
"""
SensorGuard: Engine-level risk-coverage analysis.

Key design:
- The engine is the unit of selection and resampling.
- Each engine is summarized using its mean uncertainty and its
  window-level confusion counts.
- Uncertainty-based selection is compared with random engine selection.
- Bootstrap replicates resample whole engines and recompute the ranking.
- Existing prediction CSVs are read-only.

Outputs:
  results/metrics/engine_level_risk_coverage/
    engine_level_risk_coverage.csv
    engine_level_risk_coverage_bootstrap.csv
    engine_level_summary.csv
    engine_level_input_audit.csv
    engine_level_risk_coverage.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
INPUT_FILENAME = "degradation_uncertainty_predictions.csv"

OUT = ROOT / "results" / "metrics" / "engine_level_risk_coverage"

UNCERTAINTY_COLUMNS = {
    "mutual_information": "mutual_information",
    "predictive_variance": "predictive_variance",
    "predictive_entropy": "predictive_entropy",
}

COVERAGES = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

BOOTSTRAP_REPEATS = 500
RANDOM_REPEATS = 500
SEED = 42

ENGINE_ID_CANDIDATES = [
    "engine_id",
    "unit_id",
    "unit",
    "engine",
    "unit_number",
]


def find_engine_column(df):
    for col in ENGINE_ID_CANDIDATES:
        if col in df.columns:
            return col
    return None


def load_predictions(path, subset):
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

    engine_col = find_engine_column(df)
    if engine_col is None:
        raise ValueError(
            "No engine ID column found. Engine-level analysis "
            "cannot safely proceed without engine identifiers."
        )

    numeric_cols = [
        "y_true",
        "y_pred",
        *UNCERTAINTY_COLUMNS.values(),
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[engine_col, "scenario", *numeric_cols]
    ).copy()

    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = df["y_pred"].astype(int)

    if not set(df["y_true"].unique()).issubset({0, 1}):
        raise ValueError("y_true must contain only 0 and 1.")
    if not set(df["y_pred"].unique()).issubset({0, 1}):
        raise ValueError("y_pred must contain only 0 and 1.")

    for col in UNCERTAINTY_COLUMNS.values():
        if (df[col] < 0).any():
            raise ValueError(f"Negative uncertainty in {col}.")

    df["prediction_error"] = (
        df["y_true"] != df["y_pred"]
    ).astype(int)

    df["subset"] = subset
    df["engine_id"] = df[engine_col].astype(str)

    return df, engine_col


def aggregate_engines(group):
    """
    One row per engine.

    Risk is computed as the mean error rate across retained engine windows.
    Classification metrics use pooled confusion counts across retained
    engines, so both engine-level and window-level quantities are explicit.
    """
    aggregations = {
        "n_windows": ("prediction_error", "size"),
        "engine_error_rate": ("prediction_error", "mean"),
        "actual_positive_count": ("y_true", "sum"),
        "predicted_positive_count": ("y_pred", "sum"),
    }

    engines = (
        group.groupby("engine_id", sort=False)
        .agg(**aggregations)
        .reset_index()
    )

    confusion = group.assign(
        tp=((group["y_true"] == 1) & (group["y_pred"] == 1)).astype(int),
        fp=((group["y_true"] == 0) & (group["y_pred"] == 1)).astype(int),
        tn=((group["y_true"] == 0) & (group["y_pred"] == 0)).astype(int),
        fn=((group["y_true"] == 1) & (group["y_pred"] == 0)).astype(int),
    ).groupby("engine_id", sort=False).agg(
        tp=("tp", "sum"),
        fp=("fp", "sum"),
        tn=("tn", "sum"),
        fn=("fn", "sum"),
    ).reset_index()

    engines = engines.merge(confusion, on="engine_id", how="left")

    for measure, column in UNCERTAINTY_COLUMNS.items():
        mean_scores = (
            group.groupby("engine_id", sort=False)[column]
            .mean()
            .rename(f"mean_{measure}")
            .reset_index()
        )
        engines = engines.merge(
            mean_scores, on="engine_id", how="left"
        )

    return engines


def pooled_metrics(engine_rows):
    """Compute metrics from pooled confusion counts."""
    n_engines = len(engine_rows)
    if n_engines == 0:
        return {
            "retained_engines": 0,
            "retained_windows": 0,
            "mean_engine_error_rate": np.nan,
            "pooled_window_error_rate": np.nan,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "actual_positive_rate": np.nan,
            "predicted_positive_rate": np.nan,
        }

    windows = int(engine_rows["n_windows"].sum())
    errors = float(
        (
            engine_rows["engine_error_rate"]
            * engine_rows["n_windows"]
        ).sum()
    )

    tp = int(engine_rows["tp"].sum())
    fp = int(engine_rows["fp"].sum())
    tn = int(engine_rows["tn"].sum())
    fn = int(engine_rows["fn"].sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    actual_pos = tp + fn
    predicted_pos = tp + fp

    return {
        "retained_engines": int(n_engines),
        "retained_windows": windows,
        "mean_engine_error_rate": float(
            engine_rows["engine_error_rate"].mean()
        ),
        "pooled_window_error_rate": (
            errors / windows if windows else np.nan
        ),
        "accuracy": (tp + tn) / windows if windows else np.nan,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "actual_positive_rate": actual_pos / windows if windows else np.nan,
        "predicted_positive_rate": predicted_pos / windows if windows else np.nan,
    }


def select_engine_rows(engines, score_column, coverage):
    """Retain the lowest-uncertainty engines."""
    n = len(engines)
    k = max(1, int(np.ceil(coverage * n)))

    ranked = engines.sort_values(
        score_column, ascending=True, kind="mergesort"
    )

    return ranked.iloc[:k].copy()


def bootstrap_risk(engines, score_column, coverage, repeats, rng):
    """
    Cluster bootstrap: resample engines with replacement and rerank
    by mean engine uncertainty within every replicate.

    Returns percentile intervals for pooled window error and mean
    engine error. Duplicate engine draws are preserved as repeated rows.
    """
    n = len(engines)
    if n < 2:
        return {
            "window_error_ci_low": np.nan,
            "window_error_ci_high": np.nan,
            "engine_error_ci_low": np.nan,
            "engine_error_ci_high": np.nan,
        }

    values = engines.reset_index(drop=True)
    window_errors = []
    engine_errors = []

    for _ in range(repeats):
        sample_idx = rng.integers(0, n, size=n)
        sample = values.iloc[sample_idx].reset_index(drop=True)

        selected = select_engine_rows(sample, score_column, coverage)
        metrics = pooled_metrics(selected)

        window_errors.append(metrics["pooled_window_error_rate"])
        engine_errors.append(metrics["mean_engine_error_rate"])

    wlo, whi = np.quantile(window_errors, [0.025, 0.975])
    elo, ehi = np.quantile(engine_errors, [0.025, 0.975])

    return {
        "window_error_ci_low": float(wlo),
        "window_error_ci_high": float(whi),
        "engine_error_ci_low": float(elo),
        "engine_error_ci_high": float(ehi),
    }


def random_engine_baseline(engines, coverage, repeats, rng):
    """
    Randomly select the same number of engines, with no uncertainty ranking.
    Random selection occurs at the engine level, not the window level.
    """
    n = len(engines)
    k = max(1, int(np.ceil(coverage * n)))

    if k >= n:
        metrics = pooled_metrics(engines)
        risk = metrics["pooled_window_error_rate"]
        engine_risk = metrics["mean_engine_error_rate"]
        return {
            "random_window_error_mean": risk,
            "random_window_error_ci_low": risk,
            "random_window_error_ci_high": risk,
            "random_engine_error_mean": engine_risk,
            "random_engine_error_ci_low": engine_risk,
            "random_engine_error_ci_high": engine_risk,
        }

    window_risks = []
    engine_risks = []

    for _ in range(repeats):
        idx = rng.choice(n, size=k, replace=False)
        selected = engines.iloc[idx]
        metrics = pooled_metrics(selected)
        window_risks.append(metrics["pooled_window_error_rate"])
        engine_risks.append(metrics["mean_engine_error_rate"])

    wlo, whi = np.quantile(window_risks, [0.025, 0.975])
    elo, ehi = np.quantile(engine_risks, [0.025, 0.975])

    return {
        "random_window_error_mean": float(np.mean(window_risks)),
        "random_window_error_ci_low": float(wlo),
        "random_window_error_ci_high": float(whi),
        "random_engine_error_mean": float(np.mean(engine_risks)),
        "random_engine_error_ci_low": float(elo),
        "random_engine_error_ci_high": float(ehi),
    }


def analyze_group(group, subset, scenario, rng):
    engines = aggregate_engines(group)
    n_engines = len(engines)
    baseline_metrics = pooled_metrics(engines)

    rows = []

    for measure in UNCERTAINTY_COLUMNS:
        score_column = f"mean_{measure}"

        for coverage in COVERAGES:
            selected = select_engine_rows(
                engines, score_column, coverage
            )
            selected_metrics = pooled_metrics(selected)

            boot = bootstrap_risk(
                engines,
                score_column,
                coverage,
                BOOTSTRAP_REPEATS,
                rng,
            )

            random_baseline = random_engine_baseline(
                engines,
                coverage,
                RANDOM_REPEATS,
                rng,
            )

            rows.append({
                "subset": subset,
                "scenario": scenario,
                "uncertainty_measure": measure,
                "requested_engine_coverage": coverage,
                "actual_engine_coverage": (
                    len(selected) / n_engines if n_engines else np.nan
                ),
                "total_engines": n_engines,
                "baseline_window_error_rate": (
                    baseline_metrics["pooled_window_error_rate"]
                ),
                "baseline_mean_engine_error_rate": (
                    baseline_metrics["mean_engine_error_rate"]
                ),
                **selected_metrics,
                **{
                    f"bootstrap_{key}": value
                    for key, value in boot.items()
                },
                **random_baseline,
                "window_error_reduction_vs_random": (
                    random_baseline["random_window_error_mean"]
                    - selected_metrics["pooled_window_error_rate"]
                ),
                "window_error_reduction_vs_full": (
                    baseline_metrics["pooled_window_error_rate"]
                    - selected_metrics["pooled_window_error_rate"]
                ),
            })

    return engines, rows


def make_plot(results, path):
    subsets = [s for s in SUBSETS if s in results["subset"].unique()]
    fig, axes = plt.subplots(
        len(subsets), 1,
        figsize=(11, max(4, 3.5 * len(subsets))),
        squeeze=False,
    )

    for i, subset in enumerate(subsets):
        ax = axes[i][0]
        data = results[results["subset"] == subset]

        for measure in UNCERTAINTY_COLUMNS:
            d = data[data["uncertainty_measure"] == measure]
            curve = (
                d.groupby("requested_engine_coverage", as_index=False)
                ["pooled_window_error_rate"]
                .mean()
                .sort_values("requested_engine_coverage")
            )
            ax.plot(
                curve["requested_engine_coverage"],
                curve["pooled_window_error_rate"],
                marker="o",
                markersize=3,
                label=f"{measure} selection",
            )

        random_curve = (
            data.groupby("requested_engine_coverage", as_index=False)
            ["random_window_error_mean"]
            .mean()
            .sort_values("requested_engine_coverage")
        )
        ax.plot(
            random_curve["requested_engine_coverage"],
            random_curve["random_window_error_mean"],
            linestyle="--",
            label="Random engine selection",
        )

        ax.set_title(f"{subset}: engine-level risk–coverage")
        ax.set_xlabel("Fraction of engines retained")
        ax.set_ylabel("Pooled window error rate")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    audit_rows = []
    result_rows = []
    engine_summary_rows = []

    for subset in SUBSETS:
        path = (
            ROOT
            / "results"
            / "metrics"
            / subset
            / "mc_dropout_experiments"
            / INPUT_FILENAME
        )

        if not path.exists():
            print(f"[MISSING] {subset}: {path}")
            audit_rows.append({
                "subset": subset,
                "status": "missing",
                "path": str(path),
            })
            continue

        try:
            df, original_engine_col = load_predictions(path, subset)
            print(
                f"[OK] {subset}: {len(df):,} windows, "
                f"{df['engine_id'].nunique()} engines, "
                f"ID column={original_engine_col}"
            )
            audit_rows.append({
                "subset": subset,
                "status": "loaded",
                "path": str(path),
                "rows": len(df),
                "engines": df["engine_id"].nunique(),
                "engine_id_column": original_engine_col,
            })

            for scenario, group in df.groupby("scenario", sort=False):
                engines, rows = analyze_group(
                    group.reset_index(drop=True),
                    subset,
                    scenario,
                    rng,
                )
                result_rows.extend(rows)

                for measure in UNCERTAINTY_COLUMNS:
                    score_column = f"mean_{measure}"
                    ranked = engines.sort_values(
                        score_column, ascending=True
                    )
                    engine_summary_rows.append({
                        "subset": subset,
                        "scenario": scenario,
                        "uncertainty_measure": measure,
                        "engines": len(engines),
                        "mean_engine_error_rate": (
                            engines["engine_error_rate"].mean()
                        ),
                        "median_engine_error_rate": (
                            engines["engine_error_rate"].median()
                        ),
                        "mean_engine_uncertainty": (
                            engines[score_column].mean()
                        ),
                        "median_engine_uncertainty": (
                            engines[score_column].median()
                        ),
                        "lowest_uncertainty_engine_error_rate": (
                            ranked.head(
                                max(1, int(np.ceil(0.2 * len(ranked))))
                            )["engine_error_rate"].mean()
                        ),
                    })

        except Exception as exc:
            print(f"[ERROR] {subset}: {exc}")
            audit_rows.append({
                "subset": subset,
                "status": f"failed: {exc}",
                "path": str(path),
            })

    pd.DataFrame(audit_rows).to_csv(
        OUT / "engine_level_input_audit.csv", index=False
    )

    if not result_rows:
        raise RuntimeError("No valid engine-level results were generated.")

    results = pd.DataFrame(result_rows)
    results.to_csv(
        OUT / "engine_level_risk_coverage.csv", index=False
    )

    pd.DataFrame(engine_summary_rows).to_csv(
        OUT / "engine_level_summary.csv", index=False
    )

    # Explicit bootstrap table, useful for paper reporting.
    bootstrap_columns = [
        "subset",
        "scenario",
        "uncertainty_measure",
        "requested_engine_coverage",
        "bootstrap_window_error_ci_low",
        "bootstrap_window_error_ci_high",
        "bootstrap_engine_error_ci_low",
        "bootstrap_engine_error_ci_high",
        "random_window_error_ci_low",
        "random_window_error_ci_high",
        "random_engine_error_ci_low",
        "random_engine_error_ci_high",
    ]
    results[bootstrap_columns].to_csv(
        OUT / "engine_level_risk_coverage_bootstrap.csv",
        index=False,
    )

    make_plot(
        results,
        OUT / "engine_level_risk_coverage.png",
    )

    print("\nSelected results at 20%, 50%, 80%, and 100% engine coverage:")
    selected = results[
        results["requested_engine_coverage"].isin(
            [0.20, 0.50, 0.80, 1.00]
        )
    ]

    cols = [
        "subset",
        "scenario",
        "uncertainty_measure",
        "requested_engine_coverage",
        "total_engines",
        "retained_engines",
        "pooled_window_error_rate",
        "random_window_error_mean",
        "window_error_reduction_vs_random",
        "bootstrap_window_error_ci_low",
        "bootstrap_window_error_ci_high",
    ]

    print(selected[cols].round(4).to_string(index=False))

    print("\nSaved outputs in:")
    print(OUT)
    print("\nInterpretation:")
    print("- Engines, not individual windows, are ranked for selection.")
    print("- Bootstrap replicates resample engines and recompute rankings.")
    print("- Random baseline selects engines, not individual windows.")
    print("- Window error is pooled across windows of selected engines.")
    print("- The bootstrap interval reflects sampling variability in engines.")
    print("- Test-set results remain descriptive until a policy is selected")
    print("  using validation engines and evaluated on held-out test engines.")


if __name__ == "__main__":
    main()
