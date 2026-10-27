"""
SensorGuard — RQ2 uncertainty/error reliability analysis

Reads existing rq2_paired_predictions.csv files for FD001–FD004.
Does not retrain models or rerun MC Dropout.

Run from the SensorGuard project root:
    python -u -m src.evaluation.rq2_uncertainty_error_analysis
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "metrics" / "rq2_controlled_sensor_fault_experiment"
OUTPUT_DIR = RESULTS_ROOT / "uncertainty_error_analysis"
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

# The current paired-prediction CSV exports per-window MI and variance,
# but does not export per-window predictive entropy.
UNCERTAINTY_COLUMNS = {
    "mutual_information": {
        "clean": "clean_mutual_information",
        "scenario": "scenario_mutual_information",
    },
    "predictive_variance": {
        "clean": "clean_variance",
        "scenario": "scenario_variance",
    },
}


def safe_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_auprc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(average_precision_score(labels, scores))


def summarize_metric(df, subset, scenario, metric_name, uncertainty_col, error_col):
    scores = pd.to_numeric(df[uncertainty_col], errors="coerce").to_numpy()
    errors = df[error_col].astype(bool).to_numpy().astype(int)
    valid = np.isfinite(scores)
    scores, errors = scores[valid], errors[valid]

    n = len(errors)
    n_errors = int(errors.sum())
    n_correct = n - n_errors
    mean_error = float(scores[errors == 1].mean()) if n_errors else float("nan")
    mean_correct = float(scores[errors == 0].mean()) if n_correct else float("nan")

    return {
        "subset": subset,
        "scenario": scenario,
        "uncertainty_metric": metric_name,
        "n_windows": n,
        "n_errors": n_errors,
        "error_rate": float(errors.mean()) if n else float("nan"),
        "error_detection_auroc": safe_auc(errors, scores),
        "error_detection_auprc": safe_auprc(errors, scores),
        "mean_uncertainty_on_errors": mean_error,
        "mean_uncertainty_on_correct": mean_correct,
        "mean_uncertainty_difference_error_minus_correct": (
            mean_error - mean_correct
            if np.isfinite(mean_error) and np.isfinite(mean_correct)
            else float("nan")
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    missing = []

    for subset in SUBSETS:
        input_path = RESULTS_ROOT / subset / "rq2_paired_predictions.csv"
        if not input_path.exists():
            missing.append(str(input_path))
            continue

        print(f"\nReading {subset}: {input_path}", flush=True)
        df = pd.read_csv(input_path)
        required = {
            "scenario", "clean_error", "scenario_error",
            *[col for pair in UNCERTAINTY_COLUMNS.values() for col in pair.values()],
        }
        absent = sorted(required - set(df.columns))
        if absent:
            raise ValueError(f"{subset} CSV is missing required columns: {absent}")

        for scenario, scenario_df in df.groupby("scenario", sort=False):
            for metric_name, cols in UNCERTAINTY_COLUMNS.items():
                # Clean predictions are repeated in each paired scenario;
                # evaluate them once, using the clean scenario rows.
                if scenario == "clean":
                    all_rows.append(
                        summarize_metric(
                            scenario_df, subset, "clean", metric_name,
                            cols["clean"], "clean_error"
                        )
                    )

                # Evaluate the scenario's uncertainty against errors made
                # under that same input condition.
                all_rows.append(
                    summarize_metric(
                        scenario_df, subset, scenario, metric_name,
                        cols["scenario"], "scenario_error"
                    )
                )

        print(
            f"Loaded {len(df):,} paired rows; "
            f"{df['scenario'].nunique()} scenarios.",
            flush=True,
        )

    if missing:
        raise FileNotFoundError("Missing paired-prediction files:\n" + "\n".join(missing))

    result = pd.DataFrame(all_rows).drop_duplicates(
        subset=["subset", "scenario", "uncertainty_metric"], keep="first"
    )
    result_path = OUTPUT_DIR / "rq2_uncertainty_error_metrics.csv"
    result.to_csv(result_path, index=False)

    pivot = result.pivot_table(
        index=["subset", "scenario"],
        columns="uncertainty_metric",
        values="error_detection_auroc",
        aggfunc="first",
    ).reset_index()
    pivot_path = OUTPUT_DIR / "rq2_error_detection_auroc_comparison.csv"
    pivot.to_csv(pivot_path, index=False)

    report = {
        "analysis": "uncertainty-based error detection",
        "subsets": SUBSETS,
        "uncertainty_metrics_evaluated": list(UNCERTAINTY_COLUMNS),
        "interpretation": {
            "error_detection_auroc": (
                "How well uncertainty ranks incorrect predictions above correct ones; "
                "0.5 indicates chance-level ranking."
            ),
            "error_detection_auprc": (
                "Average precision for error detection; compare with each scenario's "
                "error prevalence."
            ),
        },
        "limitations": [
            "Window-level observations from the same engine are correlated; results "
            "are descriptive and do not include engine-clustered confidence intervals.",
            "Per-window predictive entropy was not exported by the existing experiment, "
            "so entropy-based error detection cannot be calculated from these CSVs alone.",
            "Synthetic faults do not establish real-world sensor-fault diagnosis or "
            "causal attribution.",
        ],
        "metrics_csv": str(result_path),
        "auroc_comparison_csv": str(pivot_path),
    }
    report_path = OUTPUT_DIR / "rq2_uncertainty_error_analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 65)
    print("RQ2 UNCERTAINTY/ERROR ANALYSIS COMPLETED")
    print("=" * 65)
    print(f"Detailed metrics: {result_path}")
    print(f"AUROC comparison: {pivot_path}")
    print(f"Analysis report:  {report_path}")
    print("\nLowest error-detection AUROC values (descriptive):")
    print(
        result.sort_values("error_detection_auroc", na_position="last")
        .head(12)[[
            "subset", "scenario", "uncertainty_metric",
            "error_rate", "error_detection_auroc", "error_detection_auprc"
        ]].to_string(index=False)
    )
    print("\nNote: entropy is omitted because per-window entropy was not saved.")


if __name__ == "__main__":
    main()
