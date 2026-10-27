"""
SensorGuard — Engine-clustered bootstrap for RQ2 uncertainty error detection.

Reads the existing per-window paired prediction CSVs for FD001–FD004.
Compares error-detection AUROC for predictive variance vs mutual information
using paired engine-cluster bootstrap resampling.

Run from the SensorGuard project root:
    python -u -m src.evaluation.rq2_engine_cluster_bootstrap

This script does not retrain models or rerun MC Dropout.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "rq2_controlled_sensor_fault_experiment"
)
OUTPUT_DIR = RESULTS_ROOT / "uncertainty_error_analysis" / "engine_cluster_bootstrap"
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

N_BOOTSTRAPS = 2000
RANDOM_SEED = 20261027
MIN_VALID_BOOTSTRAPS = 200

# Score column, error-label column. Higher uncertainty is treated as a score
# for predicting an error (error=1); AUROC < 0.5 means the ranking is reversed.
METRICS = {
    "mutual_information": ("scenario_mutual_information", "scenario_error"),
    "predictive_variance": ("scenario_variance", "scenario_error"),
}


def auc_or_nan(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def bh_fdr(p_values):
    """Benjamini-Hochberg adjusted p-values; NaNs remain NaN."""
    p_values = np.asarray(p_values, dtype=float)
    out = np.full(p_values.shape, np.nan, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(p_values))
    if not len(valid_idx):
        return out
    vals = p_values[valid_idx]
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty(m, dtype=float)
    restored[order] = adjusted
    out[valid_idx] = restored
    return out


def bootstrap_one_condition(df, subset, scenario, rng):
    required = {
        "engine_id",
        "scenario",
        "scenario_error",
        "scenario_mutual_information",
        "scenario_variance",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns required for bootstrap: {missing}")

    # Keep the selected condition only.
    cond = df.loc[df["scenario"].astype(str) == str(scenario)].copy()
    if cond.empty:
        return None

    cond["scenario_error"] = cond["scenario_error"].astype(bool).astype(int)
    cond["scenario_mutual_information"] = pd.to_numeric(
        cond["scenario_mutual_information"], errors="coerce"
    )
    cond["scenario_variance"] = pd.to_numeric(
        cond["scenario_variance"], errors="coerce"
    )

    engine_ids = cond["engine_id"].drop_duplicates().to_numpy()
    if len(engine_ids) < 2:
        return None

    labels = cond["scenario_error"].to_numpy(dtype=int)
    mi = cond["scenario_mutual_information"].to_numpy(dtype=float)
    var = cond["scenario_variance"].to_numpy(dtype=float)

    point_mi = auc_or_nan(labels, mi)
    point_var = auc_or_nan(labels, var)
    point_diff = point_var - point_mi if np.isfinite(point_var) and np.isfinite(point_mi) else np.nan

    # Store row indices per engine, then resample engines as clusters. Sampling
    # an engine twice duplicates all its windows twice, as cluster bootstrap requires.
    rows_by_engine = {
        engine: np.flatnonzero(cond["engine_id"].to_numpy() == engine)
        for engine in engine_ids
    }

    boot_mi = []
    boot_var = []
    boot_diff = []

    for _ in range(N_BOOTSTRAPS):
        sampled_engines = rng.choice(engine_ids, size=len(engine_ids), replace=True)
        sampled_rows = np.concatenate([rows_by_engine[e] for e in sampled_engines])

        b_labels = labels[sampled_rows]
        if len(np.unique(b_labels)) < 2:
            continue

        b_mi = auc_or_nan(b_labels, mi[sampled_rows])
        b_var = auc_or_nan(b_labels, var[sampled_rows])
        if np.isfinite(b_mi) and np.isfinite(b_var):
            boot_mi.append(b_mi)
            boot_var.append(b_var)
            boot_diff.append(b_var - b_mi)

    boot_mi = np.asarray(boot_mi, dtype=float)
    boot_var = np.asarray(boot_var, dtype=float)
    boot_diff = np.asarray(boot_diff, dtype=float)

    if len(boot_diff) >= MIN_VALID_BOOTSTRAPS:
        mi_ci = np.percentile(boot_mi, [2.5, 97.5])
        var_ci = np.percentile(boot_var, [2.5, 97.5])
        diff_ci = np.percentile(boot_diff, [2.5, 97.5])
        # Two-sided bootstrap sign-tail estimate, with a plus-one correction.
        p_raw = min(
            1.0,
            2.0 * min(
                (np.sum(boot_diff <= 0) + 1) / (len(boot_diff) + 1),
                (np.sum(boot_diff >= 0) + 1) / (len(boot_diff) + 1),
            ),
        )
    else:
        mi_ci = [np.nan, np.nan]
        var_ci = [np.nan, np.nan]
        diff_ci = [np.nan, np.nan]
        p_raw = np.nan

    return {
        "subset": subset,
        "scenario": scenario,
        "n_engines": int(len(engine_ids)),
        "n_windows": int(len(cond)),
        "n_errors": int(cond["scenario_error"].sum()),
        "error_rate": float(cond["scenario_error"].mean()),
        "mi_error_detection_auroc": point_mi,
        "mi_auroc_ci95_low": float(mi_ci[0]),
        "mi_auroc_ci95_high": float(mi_ci[1]),
        "variance_error_detection_auroc": point_var,
        "variance_auroc_ci95_low": float(var_ci[0]),
        "variance_auroc_ci95_high": float(var_ci[1]),
        "variance_minus_mi_auroc": point_diff,
        "difference_ci95_low": float(diff_ci[0]),
        "difference_ci95_high": float(diff_ci[1]),
        "bootstrap_p_value_uncorrected": float(p_raw),
        "valid_bootstrap_replicates": int(len(boot_diff)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    results = []

    for subset in SUBSETS:
        input_path = RESULTS_ROOT / subset / "rq2_paired_predictions.csv"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input file: {input_path}")

        print(f"\nLoading {subset}: {input_path}", flush=True)
        df = pd.read_csv(input_path)
        if "scenario" not in df.columns:
            raise ValueError(f"{input_path} does not contain a 'scenario' column.")

        scenarios = list(df["scenario"].drop_duplicates())
        print(
            f"{subset}: {len(scenarios)} scenarios, "
            f"{df['engine_id'].nunique()} engines, {len(df):,} paired rows",
            flush=True,
        )

        for scenario in scenarios:
            print(f"  Bootstrapping {subset} / {scenario}...", flush=True)
            row = bootstrap_one_condition(df, subset, scenario, rng)
            if row is not None:
                results.append(row)

    result = pd.DataFrame(results)
    if result.empty:
        raise RuntimeError("No bootstrap results were produced.")

    # Correct the 52 condition-wise tests as one family. This is exploratory:
    # the bootstrap sign-tail p-values are approximate, not exact randomization tests.
    result["bootstrap_p_value_bh_fdr"] = bh_fdr(
        result["bootstrap_p_value_uncorrected"].to_numpy()
    )

    output_csv = OUTPUT_DIR / "rq2_engine_cluster_bootstrap_results.csv"
    result.to_csv(output_csv, index=False)

    report = {
        "analysis": "paired engine-cluster bootstrap for uncertainty error-detection AUROC",
        "subsets": SUBSETS,
        "n_bootstraps_requested_per_condition": N_BOOTSTRAPS,
        "random_seed": RANDOM_SEED,
        "comparison": "predictive variance AUROC minus mutual-information AUROC",
        "confidence_intervals": "percentile 95% intervals from engine-cluster bootstrap",
        "multiple_comparisons": (
            "Benjamini-Hochberg FDR-adjusted approximate bootstrap sign-tail p-values "
            "across all dataset-condition comparisons."
        ),
        "limitations": [
            "Bootstrap resamples engines as clusters to account for within-engine "
            "window dependence, but the estimates remain specific to these test sets.",
            "The approximate bootstrap sign-tail p-value is not an exact paired "
            "randomization test.",
            "Intervals may be unstable if there are few engines or few errors; inspect "
            "valid_bootstrap_replicates and error counts.",
            "Synthetic sensor faults do not establish real-world sensor-fault diagnosis "
            "or causal attribution.",
        ],
        "output_csv": str(output_csv),
    }
    report_path = OUTPUT_DIR / "rq2_engine_cluster_bootstrap_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("RQ2 ENGINE-CLUSTERED BOOTSTRAP COMPLETED")
    print("=" * 70)
    print(f"Results CSV: {output_csv}")
    print(f"Report JSON: {report_path}")
    print("\nLargest positive variance-minus-MI differences:")
    print(
        result.sort_values("variance_minus_mi_auroc", ascending=False)
        .head(10)[[
            "subset", "scenario", "n_engines", "error_rate",
            "mi_error_detection_auroc", "variance_error_detection_auroc",
            "variance_minus_mi_auroc", "difference_ci95_low",
            "difference_ci95_high", "bootstrap_p_value_bh_fdr",
        ]].to_string(index=False)
    )
    print("\nCases where the 95% difference interval excludes zero:")
    excludes_zero = (
        (result["difference_ci95_low"] > 0)
        | (result["difference_ci95_high"] < 0)
    )
    print(f"{int(excludes_zero.sum())} of {len(result)} conditions.")
    print(
        "\nInterpretation: positive variance-minus-MI values favor predictive variance. "
        "Use the confidence intervals and FDR-adjusted values together; do not infer "
        "significance from the point estimates alone."
    )


if __name__ == "__main__":
    main()
