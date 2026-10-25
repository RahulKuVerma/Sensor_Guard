
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SensorGuard: Corrected Paired Engine-Level Bootstrap
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

POLICY_DIR = ROOT / "results" / "metrics" / "validation_selected_rejection"
PREDICTION_DIR = ROOT / "results" / "metrics"

# New output directory: preserve all previous experiments.
OUTPUT_DIR = ROOT / "results" / "metrics" / "paired_engine_bootstrap_corrected"

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

SCENARIOS = [
    "clean",
    "random_dropout",
    "missing_interval",
    "gaussian_noise",
    "positive_bias",
    "positive_drift",
]

N_BOOTSTRAPS = 2000
RANDOM_SELECTIONS_PER_BOOTSTRAP = 50
POINT_ESTIMATE_RANDOM_SELECTIONS = 1000
RANDOM_SEED = 42

UNCERTAINTY_COLUMNS = {
    "mutual_information": "mutual_information",
    "predictive_variance": "predictive_variance",
    "predictive_entropy": "predictive_entropy",
}


def find_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Cannot find {description}. Tried {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def load_frozen_policies():
    path = POLICY_DIR / "selected_validation_policies.csv"

    if not path.exists():
        raise FileNotFoundError(f"Frozen policy file not found: {path}")

    policies = pd.read_csv(path)

    subset_col = find_column(
        policies, ["subset", "dataset"], "subset column"
    )
    metric_col = find_column(
        policies,
        ["selected_uncertainty", "uncertainty_metric", "metric"],
        "selected uncertainty column",
    )
    threshold_col = find_column(
        policies,
        ["threshold", "selected_threshold", "uncertainty_threshold"],
        "threshold column",
    )

    policies = policies.rename(
        columns={
            subset_col: "subset",
            metric_col: "selected_uncertainty",
            threshold_col: "threshold",
        }
    )

    policies["subset"] = policies["subset"].astype(str).str.upper()
    policies["selected_uncertainty"] = (
        policies["selected_uncertainty"].astype(str).str.lower().str.strip()
    )

    if policies["subset"].duplicated().any():
        raise ValueError("More than one frozen policy exists for a subset.")

    return policies.set_index("subset")


def load_predictions(subset, scenario):
    path = (
        PREDICTION_DIR
        / subset
        / "mc_dropout_experiments"
        / "degradation_uncertainty_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path)

    required = {
        "scenario", "engine_id", "y_true", "y_pred"
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    df = df[df["scenario"].astype(str) == scenario].copy()

    if df.empty:
        raise ValueError(f"No predictions for {subset}/{scenario}")

    return df


def aggregate_engines(df, uncertainty_column):
    """
    Create one record per engine.

    Each engine retains its window count and error count. Risk is
    calculated as total errors / total windows among retained engines.
    """
    if uncertainty_column not in df.columns:
        raise ValueError(
            f"Missing uncertainty column '{uncertainty_column}'. "
            f"Available columns: {list(df.columns)}"
        )

    data = df.copy()

    for col in ["y_true", "y_pred", uncertainty_column]:
        data[col] = pd.to_numeric(data[col], errors="raise")

    if data[["y_true", "y_pred", uncertainty_column]].isna().any().any():
        raise ValueError("Predictions contain missing required values.")

    data["window_error"] = (
        data["y_true"].astype(int) != data["y_pred"].astype(int)
    ).astype(np.int64)

    engines = (
        data.groupby("engine_id", as_index=False)
        .agg(
            uncertainty_score=(uncertainty_column, "mean"),
            n_windows=("window_error", "size"),
            n_errors=("window_error", "sum"),
        )
    )

    if len(engines) < 2:
        raise ValueError("At least two engines are required.")

    if (engines["n_windows"] <= 0).any():
        raise ValueError("An engine has no prediction windows.")

    return engines


def risk_from_indices(errors, windows, indices):
    """Window-weighted error rate for selected engine-row indices."""
    if len(indices) == 0:
        return np.nan

    denominator = windows[indices].sum()

    if denominator <= 0:
        return np.nan

    return float(errors[indices].sum() / denominator)


def random_matched_risks(errors, windows, k, repeats, rng):
    """
    Draw random subsets of exactly k engine instances.

    Engine instances are sampled without replacement within each draw.
    Bootstrap duplicates, if present, represent separate sampled clusters.
    """
    n = len(errors)

    if k == 0:
        return np.full(repeats, np.nan)

    if k >= n:
        # If every engine is retained, random and policy risks are identical.
        all_indices = np.arange(n)
        risk = risk_from_indices(errors, windows, all_indices)
        return np.full(repeats, risk)

    output = np.empty(repeats, dtype=float)

    for r in range(repeats):
        indices = rng.choice(n, size=k, replace=False)
        output[r] = risk_from_indices(errors, windows, indices)

    return output


def paired_bootstrap(engines, threshold, rng):
    """
    Estimate policy risk minus random risk at matched engine coverage.

    The same bootstrap sample is used for both methods. The threshold is
    fixed before test evaluation and is never refitted on test data.
    """
    scores = engines["uncertainty_score"].to_numpy(dtype=float)
    errors = engines["n_errors"].to_numpy(dtype=float)
    windows = engines["n_windows"].to_numpy(dtype=float)

    n = len(engines)

    # Frozen validation-selected policy.
    retained_mask = scores <= threshold
    retained_indices = np.flatnonzero(retained_mask)
    k = len(retained_indices)

    if k == 0:
        raise ValueError(
            "Frozen threshold rejects every engine; cannot compare risks."
        )

    observed_policy_risk = risk_from_indices(
        errors, windows, retained_indices
    )

    observed_random_risks = random_matched_risks(
        errors,
        windows,
        k,
        POINT_ESTIMATE_RANDOM_SELECTIONS,
        rng,
    )
    observed_random_risk = float(np.nanmean(observed_random_risks))

    observed_difference = observed_policy_risk - observed_random_risk

    # Bootstrap resamples whole engine records, not individual windows.
    bootstrap_differences = []
    bootstrap_policy_risks = []
    bootstrap_random_risks = []
    bootstrap_coverages = []

    for _ in range(N_BOOTSTRAPS):
        sampled_rows = rng.integers(0, n, size=n)

        sample_scores = scores[sampled_rows]
        sample_errors = errors[sampled_rows]
        sample_windows = windows[sampled_rows]

        # Keep the original validation-selected threshold fixed.
        sample_mask = sample_scores <= threshold
        selected = np.flatnonzero(sample_mask)
        sample_k = len(selected)

        if sample_k == 0:
            # Undefined policy risk in this replicate.
            continue

        policy_risk = risk_from_indices(
            sample_errors, sample_windows, selected
        )

        random_risks = random_matched_risks(
            sample_errors,
            sample_windows,
            sample_k,
            RANDOM_SELECTIONS_PER_BOOTSTRAP,
            rng,
        )

        random_risk = float(np.nanmean(random_risks))

        if not (
            np.isfinite(policy_risk)
            and np.isfinite(random_risk)
        ):
            continue

        # Both risks use the same sampled engines and matched engine count.
        difference = policy_risk - random_risk

        bootstrap_policy_risks.append(policy_risk)
        bootstrap_random_risks.append(random_risk)
        bootstrap_differences.append(difference)
        bootstrap_coverages.append(sample_k / n)

    differences = np.asarray(bootstrap_differences, dtype=float)

    if len(differences) < 500:
        raise RuntimeError(
            f"Only {len(differences)} valid bootstrap replicates. "
            "Check the frozen threshold and engine data."
        )

    ci_low, ci_high = np.quantile(differences, [0.025, 0.975])

    # Descriptive bootstrap sign probability, not an exact hypothesis test.
    p_left = (np.sum(differences <= 0) + 1) / (len(differences) + 1)
    p_right = (np.sum(differences >= 0) + 1) / (len(differences) + 1)
    approximate_p = min(1.0, 2.0 * min(p_left, p_right))

    random_full_risk = risk_from_indices(
        errors, windows, np.arange(n)
    )

    return {
        "n_engines": n,
        "n_retained_engines": k,
        "actual_engine_coverage": k / n,
        "target_coverage": 0.80,
        "frozen_threshold": float(threshold),
        "uncertainty_policy_error": observed_policy_risk,
        "random_matched_error": observed_random_risk,
        "full_test_error": random_full_risk,
        "error_difference_policy_minus_random": observed_difference,
        "error_reduction_vs_random": -observed_difference,
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "bootstrap_p_value_approx": float(approximate_p),
        "bootstrap_replicates_requested": N_BOOTSTRAPS,
        "bootstrap_replicates_used": int(len(differences)),
        "mean_bootstrap_coverage": float(np.mean(bootstrap_coverages)),
        "bootstrap_mean_policy_error": float(np.mean(bootstrap_policy_risks)),
        "bootstrap_mean_random_error": float(np.mean(bootstrap_random_risks)),
        "ci_excludes_zero": bool(ci_high < 0 or ci_low > 0),
        "random_baseline_matches_retained_engine_count": True,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    policies = load_frozen_policies()
    rng = np.random.default_rng(RANDOM_SEED)

    records = []

    print("=" * 80)
    print("SensorGuard — Corrected Paired Engine-Level Bootstrap")
    print("=" * 80)
    print(f"Bootstrap replicates: {N_BOOTSTRAPS}")
    print(f"Random selections per bootstrap: {RANDOM_SELECTIONS_PER_BOOTSTRAP}")
    print(f"Random selections for point estimate: {POINT_ESTIMATE_RANDOM_SELECTIONS}")
    print("Policy thresholds: frozen from validation")
    print("Bootstrap unit: engine")
    print("Negative difference favors uncertainty-based rejection.")
    print("The approximate p-value is descriptive, not an exact randomization test.")

    for subset in SUBSETS:
        if subset not in policies.index:
            raise ValueError(f"No frozen validation policy for {subset}")

        metric = policies.loc[subset, "selected_uncertainty"]
        threshold = float(policies.loc[subset, "threshold"])

        if metric not in UNCERTAINTY_COLUMNS:
            raise ValueError(
                f"Unsupported uncertainty metric '{metric}' for {subset}"
            )

        uncertainty_column = UNCERTAINTY_COLUMNS[metric]

        print(
            f"\n--- {subset}: metric={metric}, "
            f"threshold={threshold:.10g} ---"
        )

        for scenario in SCENARIOS:
            predictions = load_predictions(subset, scenario)
            engines = aggregate_engines(predictions, uncertainty_column)

            result = paired_bootstrap(
                engines=engines,
                threshold=threshold,
                rng=rng,
            )

            result["subset"] = subset
            result["scenario"] = scenario
            result["selected_uncertainty"] = metric
            records.append(result)

            print(
                f"{scenario:18s} "
                f"coverage={result['actual_engine_coverage']:.3f} "
                f"policy={result['uncertainty_policy_error']:.4f} "
                f"random={result['random_matched_error']:.4f} "
                f"diff={result['error_difference_policy_minus_random']:+.4f} "
                f"CI=[{result['bootstrap_ci_low']:+.4f}, "
                f"{result['bootstrap_ci_high']:+.4f}] "
                f"p~{result['bootstrap_p_value_approx']:.4f}"
            )

    results = pd.DataFrame(records)

    results_path = OUTPUT_DIR / "paired_engine_bootstrap_results.csv"
    results.to_csv(results_path, index=False)

    summary = (
        results.groupby("subset", as_index=False)
        .agg(
            scenarios=("scenario", "count"),
            mean_error_difference=(
                "error_difference_policy_minus_random", "mean"
            ),
            mean_error_reduction=(
                "error_reduction_vs_random", "mean"
            ),
            scenarios_with_lower_policy_error=(
                "error_difference_policy_minus_random",
                lambda values: int((values < 0).sum()),
            ),
            scenarios_ci_excludes_zero=("ci_excludes_zero", "sum"),
            mean_actual_coverage=("actual_engine_coverage", "mean"),
        )
    )

    summary_path = OUTPUT_DIR / "paired_engine_bootstrap_subset_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("SUMMARY BY SUBSET")
    print("=" * 80)
    print(summary.to_string(index=False))

    print("\nSaved files:")
    print(results_path)
    print(summary_path)

    print(
        "\nInterpretation: the difference is policy error minus matched-random "
        "error. A negative difference favors uncertainty rejection. The "
        "percentile interval is conditional on the frozen policy and the "
        "observed test scenarios. The sign-based p-value is approximate; "
        "do not treat it as a definitive confirmatory test."
    )


if __name__ == "__main__":
    main()
