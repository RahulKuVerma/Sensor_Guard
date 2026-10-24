
"""
SensorGuard: Risk-coverage analysis with random-rejection baseline.

For every subset and degradation scenario:
- Retains the least-uncertain predictions at several coverage levels.
- Compares uncertainty-based rejection against random rejection.
- Reports retained error rates, positive-class recall, and F1.
- Uses engine-level bootstrap confidence intervals if an engine ID
  column is available; otherwise reports that engine-level inference
  is unavailable.

No model training, threshold fitting, or modification of source CSVs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, recall_score


ROOT = Path(__file__).resolve().parents[2]
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

INPUT_NAME = "degradation_uncertainty_predictions.csv"
INPUT_FOLDER = "mc_dropout_experiments"

OUT = ROOT / "results" / "metrics" / "risk_coverage_analysis"

COVERAGES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

UNCERTAINTY_COLUMNS = {
    "mutual_information": "mutual_information",
    "predictive_variance": "predictive_variance",
    "predictive_entropy": "predictive_entropy",
}

RANDOM_REPEATS = 200
BOOTSTRAP_REPEATS = 300
SEED = 42

# Common possible names for the engine identifier.
ENGINE_ID_CANDIDATES = [
    "engine_id",
    "unit_id",
    "unit",
    "engine",
    "unit_number",
]


def find_engine_column(df):
    """Return a recognized engine-ID column, if present."""
    for column in ENGINE_ID_CANDIDATES:
        if column in df.columns:
            return column
    return None


def retained_metrics(df):
    """Metrics for a retained set of predictions."""
    y = df["y_true"].to_numpy(dtype=int)
    pred = df["y_pred"].to_numpy(dtype=int)

    return {
        "retained_error_rate": float(np.mean(y != pred)),
        "retained_accuracy": float(np.mean(y == pred)),
        "retained_positive_recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "retained_f1": float(
            f1_score(y, pred, zero_division=0)
        ),
        "retained_positive_rate": float(np.mean(pred)),
        "retained_count": int(len(df)),
    }


def bootstrap_engine_error_ci(df, engine_col, repeats, rng):
    """
    Cluster bootstrap over engines, not individual windows.

    Returns a 95% percentile interval for the error rate of the
    supplied retained set. Multiple windows from a sampled engine
    stay together. This is a descriptive cluster-bootstrap interval.
    """
    if engine_col is None or engine_col not in df.columns:
        return np.nan, np.nan, "engine ID unavailable"

    engine_ids = df[engine_col].dropna().unique()

    if len(engine_ids) < 2:
        return np.nan, np.nan, "too few identified engines"

    grouped = {
        engine: df.loc[df[engine_col] == engine, "prediction_error"]
        .to_numpy(dtype=int)
        for engine in engine_ids
    }

    estimates = []

    for _ in range(repeats):
        sampled_ids = rng.choice(
            engine_ids, size=len(engine_ids), replace=True
        )
        sampled_errors = np.concatenate(
            [grouped[engine] for engine in sampled_ids]
        )

        if len(sampled_errors):
            estimates.append(float(sampled_errors.mean()))

    if not estimates:
        return np.nan, np.nan, "bootstrap unavailable"

    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper), "engine-cluster bootstrap"


def random_rejection_baseline(df, retain_count, repeats, rng):
    """
    Randomly retain the same number of rows, without using uncertainty.

    This gives a finite-sample reference distribution for retained error
    rate. It is a row-level random baseline, not an engine-level policy.
    """
    n = len(df)
    errors = df["prediction_error"].to_numpy(dtype=int)

    if retain_count >= n:
        value = float(errors.mean())
        return value, value, value

    values = []

    for _ in range(repeats):
        idx = rng.choice(n, size=retain_count, replace=False)
        values.append(float(errors[idx].mean()))

    lower, mean, upper = np.quantile(
        values, [0.025, 0.50, 0.975]
    )

    return float(mean), float(lower), float(upper)


def prepare_frame(path, subset):
    df = pd.read_csv(path)

    required = {
        "scenario",
        "y_true",
        "y_pred",
        *UNCERTAINTY_COLUMNS.values(),
    }
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for column in [
        "y_true",
        "y_pred",
        *UNCERTAINTY_COLUMNS.values(),
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=[
            "scenario",
            "y_true",
            "y_pred",
            *UNCERTAINTY_COLUMNS.values(),
        ]
    ).copy()

    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = df["y_pred"].astype(int)
    df["prediction_error"] = (
        df["y_true"] != df["y_pred"]
    ).astype(int)
    df["subset"] = subset

    if not set(df["y_true"].unique()).issubset({0, 1}):
        raise ValueError("y_true must be binary.")

    if not set(df["y_pred"].unique()).issubset({0, 1}):
        raise ValueError("y_pred must be binary.")

    for column in UNCERTAINTY_COLUMNS.values():
        if (df[column] < 0).any():
            raise ValueError(f"Negative uncertainty value in {column}.")

    return df


def analyze_group(group, subset, scenario, engine_col, rng):
    n = len(group)
    rows = []

    for measure, column in UNCERTAINTY_COLUMNS.items():
        ordered = group.sort_values(
            column, ascending=True, kind="mergesort"
        ).reset_index(drop=True)

        for coverage in COVERAGES:
            retain_count = max(1, int(np.ceil(coverage * n)))
            retained = ordered.iloc[:retain_count]

            metrics = retained_metrics(retained)
            random_mean, random_low, random_high = (
                random_rejection_baseline(
                    group, retain_count, RANDOM_REPEATS, rng
                )
            )

            ci_low, ci_high, ci_method = bootstrap_engine_error_ci(
                retained,
                engine_col,
                BOOTSTRAP_REPEATS,
                rng,
            )

            rows.append({
                "subset": subset,
                "scenario": scenario,
                "uncertainty_measure": measure,
                "requested_coverage": coverage,
                "actual_coverage": retain_count / n,
                "total_windows": n,
                "retained_count": retain_count,
                **metrics,
                "random_baseline_error_rate": random_mean,
                "random_baseline_error_p2_5": random_low,
                "random_baseline_error_p97_5": random_high,
                "engine_bootstrap_error_ci_low": ci_low,
                "engine_bootstrap_error_ci_high": ci_high,
                "engine_bootstrap_method": ci_method,
            })

    return rows


def plot_curves(results, filename):
    subsets = [s for s in SUBSETS if s in results["subset"].unique()]
    fig, axes = plt.subplots(
        len(subsets), 1,
        figsize=(11, max(4, 3.6 * len(subsets))),
        squeeze=False,
    )

    for i, subset in enumerate(subsets):
        ax = axes[i][0]
        data = results[results["subset"] == subset]

        for measure in UNCERTAINTY_COLUMNS:
            d = data[data["uncertainty_measure"] == measure]

            # Mean across scenarios for overview only.
            curve = (
                d.groupby("requested_coverage", as_index=False)
                .agg(
                    risk=("retained_error_rate", "mean"),
                    random_risk=("random_baseline_error_rate", "mean"),
                )
                .sort_values("requested_coverage")
            )

            ax.plot(
                curve["requested_coverage"],
                curve["risk"],
                marker="o",
                markersize=3,
                label=f"{measure} rejection",
            )

        # Same random baseline is repeated for each uncertainty measure.
        random_curve = (
            data.groupby("requested_coverage", as_index=False)
            ["random_baseline_error_rate"]
            .mean()
            .sort_values("requested_coverage")
        )

        ax.plot(
            random_curve["requested_coverage"],
            random_curve["random_baseline_error_rate"],
            linestyle="--",
            label="Random rejection",
        )

        ax.set_title(f"{subset}: mean risk-coverage overview")
        ax.set_xlabel("Fraction of predictions retained")
        ax.set_ylabel("Retained error rate")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    all_rows = []
    availability = []
    frames = []

    for subset in SUBSETS:
        path = (
            ROOT
            / "results"
            / "metrics"
            / subset
            / INPUT_FOLDER
            / INPUT_NAME
        )

        if not path.exists():
            print(f"[MISSING] {subset}: {path}")
            availability.append({
                "subset": subset,
                "status": "missing input",
                "path": str(path),
            })
            continue

        try:
            df = prepare_frame(path, subset)
            engine_col = find_engine_column(df)

            print(
                f"[OK] {subset}: {len(df):,} windows; "
                f"engine ID column: {engine_col or 'not found'}"
            )

            availability.append({
                "subset": subset,
                "status": "loaded",
                "path": str(path),
                "engine_id_column": engine_col or "",
                "rows": len(df),
            })
            frames.append((subset, df, engine_col))

        except Exception as exc:
            print(f"[ERROR] {subset}: {exc}")
            availability.append({
                "subset": subset,
                "status": f"failed: {exc}",
                "path": str(path),
            })

    pd.DataFrame(availability).to_csv(
        OUT / "input_audit.csv", index=False
    )

    if not frames:
        raise FileNotFoundError("No valid prediction files were found.")

    for subset, df, engine_col in frames:
        for scenario, group in df.groupby("scenario", sort=False):
            all_rows.extend(
                analyze_group(
                    group.reset_index(drop=True),
                    subset,
                    scenario,
                    engine_col,
                    rng,
                )
            )

    results = pd.DataFrame(all_rows)
    csv_path = OUT / "risk_coverage_with_random_baseline.csv"
    results.to_csv(csv_path, index=False)

    plot_path = OUT / "risk_coverage_random_baseline.png"
    plot_curves(results, plot_path)

    print("\nSaved:")
    print(csv_path)
    print(plot_path)
    print(OUT / "input_audit.csv")

    print("\nSelected coverage results (80%, 50%, and 20%):")
    selected = results[
        results["requested_coverage"].isin([0.20, 0.50, 0.80, 1.00])
    ]

    cols = [
        "subset",
        "scenario",
        "uncertainty_measure",
        "requested_coverage",
        "retained_error_rate",
        "random_baseline_error_rate",
        "engine_bootstrap_error_ci_low",
        "engine_bootstrap_error_ci_high",
        "engine_bootstrap_method",
    ]

    print(selected[cols].round(4).to_string(index=False))

    print("\nInterpretation reminders:")
    print("- Lower retained error than random rejection is desirable.")
    print("- Compare within the same subset, scenario, and coverage.")
    print("- Curves averaged across scenarios are descriptive only.")
    print("- The random baseline samples windows, not whole engines.")
    print("- Engine bootstrap is available only if an engine ID is saved.")
    print("- A test-set curve is not a validation-selected deployment policy.")


if __name__ == "__main__":
    main()
