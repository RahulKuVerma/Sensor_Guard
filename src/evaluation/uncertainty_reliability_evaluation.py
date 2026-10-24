
"""
SensorGuard: Consolidated uncertainty-reliability evaluation.

Reads existing MC-Dropout prediction CSVs for FD001-FD004.
Does not train models or modify existing prediction files.

Outputs:
  results/metrics/uncertainty_reliability/
    uncertainty_reliability_summary.csv
    risk_coverage_summary.csv
    missing_inputs.csv
    error_detection_auroc.png
    risk_coverage_curves.png
    prediction_collapse.png
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

INPUT_RELATIVE_PATH = Path(
    "results/metrics"
)  # Followed by subset/mc_dropout_experiments/filename

INPUT_FILENAME = "degradation_uncertainty_predictions.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "metrics" / "uncertainty_reliability"

UNCERTAINTY_COLUMNS = {
    "mutual_information": "mutual_information",
    "predictive_variance": "predictive_variance",
    "predictive_entropy": "predictive_entropy",
}

# Retain the least-uncertain predictions at each coverage level.
COVERAGE_LEVELS = [
    0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90, 1.00,
]

EPSILON = 1e-12


def safe_auc(y_true, scores):
    """Return AUROC, or NaN if the target contains only one class."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(y_true)) < 2:
        return np.nan

    if not np.isfinite(scores).all():
        return np.nan

    return float(roc_auc_score(y_true, scores))


def safe_auprc(y_true, scores):
    """Average precision for identifying incorrect predictions."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan

    if not np.isfinite(scores).all():
        return np.nan

    return float(average_precision_score(y_true, scores))


def validate_and_prepare(df, subset, path):
    """Validate required columns and standardize prediction errors."""
    required = {
        "scenario",
        "y_true",
        "y_pred",
        "mutual_information",
        "predictive_variance",
        "predictive_entropy",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}"
        )

    df = df.copy()
    df["subset"] = subset

    numeric_columns = [
        "y_true",
        "y_pred",
        *UNCERTAINTY_COLUMNS.values(),
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove only rows that cannot support the main analysis.
    before = len(df)
    df = df.dropna(
        subset=[
            "scenario",
            "y_true",
            "y_pred",
            *UNCERTAINTY_COLUMNS.values(),
        ]
    ).copy()

    if len(df) < before:
        warnings.warn(
            f"{subset}: removed {before - len(df)} rows with missing "
            "required values."
        )

    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = df["y_pred"].astype(int)

    if not set(df["y_true"].unique()).issubset({0, 1}):
        raise ValueError(f"{subset}: y_true must contain binary labels.")

    if not set(df["y_pred"].unique()).issubset({0, 1}):
        raise ValueError(f"{subset}: y_pred must contain binary predictions.")

    # Recompute the error target directly from labels and predictions.
    # This avoids depending on potentially inconsistent saved error flags.
    df["prediction_error"] = (
        df["y_true"] != df["y_pred"]
    ).astype(int)

    # Negative or non-finite uncertainty values cannot be ranked reliably.
    # Non-finite values were removed above; flag negative values explicitly.
    for column in UNCERTAINTY_COLUMNS.values():
        if (df[column] < 0).any():
            raise ValueError(
                f"{subset}: uncertainty column {column} has negative values."
            )

    # Preserve existing probability column when present.
    if "y_prob_mean" in df.columns:
        df["y_prob_mean"] = pd.to_numeric(
            df["y_prob_mean"], errors="coerce"
        )

    return df


def summarize_group(group, subset, scenario):
    """Calculate classification and uncertainty-based error-detection metrics."""
    y_true = group["y_true"].to_numpy(dtype=int)
    y_pred = group["y_pred"].to_numpy(dtype=int)
    errors = group["prediction_error"].to_numpy(dtype=int)

    row = {
        "subset": subset,
        "scenario": scenario,
        "n_windows": int(len(group)),
        "actual_positive_rate": float(y_true.mean()),
        "predicted_positive_rate": float(y_pred.mean()),
        "error_rate": float(errors.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_positives": int(((y_true == 1) & (y_pred == 1)).sum()),
        "false_positives": int(((y_true == 0) & (y_pred == 1)).sum()),
        "true_negatives": int(((y_true == 0) & (y_pred == 0)).sum()),
        "false_negatives": int(((y_true == 1) & (y_pred == 0)).sum()),
        "positive_predictions_zero": bool(y_pred.sum() == 0),
        "positive_predictions_all": bool(y_pred.sum() == len(y_pred)),
    }

    if "y_prob_mean" in group.columns:
        probs = group["y_prob_mean"].to_numpy(dtype=float)
        probs = probs[np.isfinite(probs)]
        row["mean_predicted_probability"] = (
            float(probs.mean()) if len(probs) else np.nan
        )
    else:
        row["mean_predicted_probability"] = np.nan

    # AUROC/AUPRC treat a high uncertainty score as evidence of an error.
    for metric_name, column in UNCERTAINTY_COLUMNS.items():
        scores = group[column].to_numpy(dtype=float)
        row[f"{metric_name}_error_auroc"] = safe_auc(errors, scores)
        row[f"{metric_name}_error_auprc"] = safe_auprc(errors, scores)
        row[f"mean_{metric_name}"] = float(np.mean(scores))
        row[f"median_{metric_name}"] = float(np.median(scores))

    return row


def calculate_risk_coverage(group, subset, scenario):
    """
    Sort by uncertainty from lowest to highest.
    Retain the least-uncertain predictions and calculate their error rate.
    """
    rows = []
    n = len(group)

    for metric_name, column in UNCERTAINTY_COLUMNS.items():
        ordered = group.sort_values(
            column, ascending=True, kind="mergesort"
        ).reset_index(drop=True)

        for coverage in COVERAGE_LEVELS:
            retained_count = max(1, int(np.ceil(coverage * n)))
            retained = ordered.iloc[:retained_count]

            rows.append({
                "subset": subset,
                "scenario": scenario,
                "uncertainty_measure": metric_name,
                "requested_coverage": coverage,
                "actual_coverage": len(retained) / n,
                "n_retained": len(retained),
                "retained_error_rate": float(
                    retained["prediction_error"].mean()
                ),
                "retained_accuracy": float(
                    1.0 - retained["prediction_error"].mean()
                ),
                "retained_positive_recall": float(
                    recall_score(
                        retained["y_true"],
                        retained["y_pred"],
                        zero_division=0,
                    )
                ),
            })

    return rows


def plot_error_detection(summary, output_path):
    """Plot error-detection AUROC by subset and scenario."""
    measures = [
        ("mutual_information", "Mutual information"),
        ("predictive_variance", "Predictive variance"),
        ("predictive_entropy", "Predictive entropy"),
    ]

    subsets = [s for s in SUBSETS if s in summary["subset"].unique()]
    if not subsets:
        return

    fig, axes = plt.subplots(
        len(subsets), 1,
        figsize=(13, max(4, 3.6 * len(subsets))),
        squeeze=False,
    )

    for index, subset in enumerate(subsets):
        ax = axes[index][0]
        data = summary[summary["subset"] == subset].copy()
        scenarios = list(data["scenario"].drop_duplicates())
        x = np.arange(len(scenarios))
        width = 0.24

        for measure_index, (measure, label) in enumerate(measures):
            values = []
            for scenario in scenarios:
                match = data[data["scenario"] == scenario]
                values.append(
                    match[f"{measure}_error_auroc"].iloc[0]
                )

            offsets = (
                measure_index - (len(measures) - 1) / 2
            ) * width

            ax.bar(
                x + offsets,
                values,
                width=width,
                label=label,
            )

        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Error-detection AUROC")
        ax.set_title(subset)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")

    fig.suptitle(
        "SensorGuard: Uncertainty-based prediction error detection",
        y=1.002,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_risk_coverage(risk_df, output_path):
    """Plot retained error rate against prediction coverage."""
    subsets = [s for s in SUBSETS if s in risk_df["subset"].unique()]
    if not subsets:
        return

    measures = [
        ("mutual_information", "Mutual information"),
        ("predictive_variance", "Predictive variance"),
        ("predictive_entropy", "Predictive entropy"),
    ]

    fig, axes = plt.subplots(
        len(subsets), 1,
        figsize=(10, max(4, 3.5 * len(subsets))),
        squeeze=False,
    )

    for index, subset in enumerate(subsets):
        ax = axes[index][0]
        subset_df = risk_df[risk_df["subset"] == subset]

        for measure, label in measures:
            data = subset_df[
                subset_df["uncertainty_measure"] == measure
            ]

            # Average across scenarios for a compact overview.
            # Per-scenario values remain available in the CSV.
            average_curve = (
                data.groupby("requested_coverage", as_index=False)
                ["retained_error_rate"]
                .mean()
                .sort_values("requested_coverage")
            )

            ax.plot(
                average_curve["requested_coverage"],
                average_curve["retained_error_rate"],
                marker="o",
                markersize=3,
                label=label,
            )

        ax.set_title(f"{subset}: mean risk–coverage curve across scenarios")
        ax.set_xlabel("Coverage retained (lower uncertainty first)")
        ax.set_ylabel("Retained prediction error rate")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_collapse(summary, output_path):
    """Compare actual and predicted positive rates."""
    subsets = [s for s in SUBSETS if s in summary["subset"].unique()]
    if not subsets:
        return

    fig, axes = plt.subplots(
        len(subsets), 1,
        figsize=(12, max(4, 3.5 * len(subsets))),
        squeeze=False,
    )

    for index, subset in enumerate(subsets):
        ax = axes[index][0]
        data = summary[summary["subset"] == subset].copy()
        x = np.arange(len(data))
        width = 0.36

        ax.bar(
            x - width / 2,
            data["actual_positive_rate"],
            width,
            label="Actual positive rate",
        )
        ax.bar(
            x + width / 2,
            data["predicted_positive_rate"],
            width,
            label="Predicted positive rate",
        )

        ax.set_title(f"{subset}: early-fault prediction prevalence")
        ax.set_ylabel("Positive rate")
        ax.set_xticks(x)
        ax.set_xticklabels(data["scenario"], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_frames = []
    missing_records = []

    print("=" * 72)
    print("SensorGuard — Uncertainty Reliability Evaluation")
    print("=" * 72)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")

    for subset in SUBSETS:
        input_path = (
            PROJECT_ROOT
            / INPUT_RELATIVE_PATH
            / subset
            / "mc_dropout_experiments"
            / INPUT_FILENAME
        )

        if not input_path.exists():
            print(f"[MISSING] {subset}: {input_path}")
            missing_records.append({
                "subset": subset,
                "expected_path": str(input_path),
                "status": "missing prediction CSV",
            })
            continue

        try:
            raw = pd.read_csv(input_path)
            prepared = validate_and_prepare(raw, subset, input_path)
            all_frames.append(prepared)
            print(
                f"[OK] {subset}: loaded {len(prepared):,} valid rows "
                f"from {input_path}"
            )
        except Exception as exc:
            print(f"[ERROR] {subset}: {exc}")
            missing_records.append({
                "subset": subset,
                "expected_path": str(input_path),
                "status": f"failed: {exc}",
            })

    pd.DataFrame(
        missing_records,
        columns=["subset", "expected_path", "status"],
    ).to_csv(OUTPUT_DIR / "missing_inputs.csv", index=False)

    if not all_frames:
        raise FileNotFoundError(
            "No valid prediction CSVs were found. Check the paths printed above."
        )

    predictions = pd.concat(all_frames, ignore_index=True)

    summary_rows = []
    risk_rows = []

    for (subset, scenario), group in predictions.groupby(
        ["subset", "scenario"], sort=False
    ):
        summary_rows.append(summarize_group(group, subset, scenario))
        risk_rows.extend(
            calculate_risk_coverage(group, subset, scenario)
        )

    summary_df = pd.DataFrame(summary_rows)
    risk_df = pd.DataFrame(risk_rows)

    summary_path = OUTPUT_DIR / "uncertainty_reliability_summary.csv"
    risk_path = OUTPUT_DIR / "risk_coverage_summary.csv"

    summary_df.to_csv(summary_path, index=False)
    risk_df.to_csv(risk_path, index=False)

    plot_error_detection(
        summary_df,
        OUTPUT_DIR / "error_detection_auroc.png",
    )
    plot_risk_coverage(
        risk_df,
        OUTPUT_DIR / "risk_coverage_curves.png",
    )
    plot_prediction_collapse(
        summary_df,
        OUTPUT_DIR / "prediction_collapse.png",
    )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    display_columns = [
        "subset",
        "scenario",
        "error_rate",
        "f1",
        "predicted_positive_rate",
        "mutual_information_error_auroc",
        "predictive_variance_error_auroc",
        "predictive_entropy_error_auroc",
    ]

    print(
        summary_df[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nSaved files:")
    print(f"  {summary_path}")
    print(f"  {risk_path}")
    print(f"  {OUTPUT_DIR / 'missing_inputs.csv'}")
    print(f"  {OUTPUT_DIR / 'error_detection_auroc.png'}")
    print(f"  {OUTPUT_DIR / 'risk_coverage_curves.png'}")
    print(f"  {OUTPUT_DIR / 'prediction_collapse.png'}")

    print("\nNotes:")
    print("- AUROC/AUPRC target prediction errors, not engine faults.")
    print("- Risk–coverage retains the least-uncertain predictions first.")
    print("- Risk–coverage plots average across scenarios for visualization;")
    print("  the CSV preserves separate subset/scenario/measure results.")
    print("- Sliding windows from the same engine are correlated observations.")
    print("- Missing input subsets are listed in missing_inputs.csv.")
    print("- This script evaluates existing predictions; it does not retrain.")


if __name__ == "__main__":
    main()
