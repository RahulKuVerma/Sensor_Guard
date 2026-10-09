
"""
SensorGuard - Calibration Evaluation

Reads saved MC-Dropout mean probabilities and evaluates calibration.
This script does not retrain the model or fit a calibrator.

Run from the project root:
    python -m src.evaluation.calibration_experiments
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.uncertainty.calibration import calibration_metrics


SUBSET = "FD001"
N_BINS = 15

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / SUBSET
    / "mc_dropout_experiments"
    / "degradation_uncertainty_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / SUBSET
    / "calibration"
)

SCENARIOS = [
    "clean",
    "random_dropout",
    "missing_interval",
    "gaussian_noise",
    "positive_bias",
    "positive_drift",
]


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_CSV}\n"
            "Run mc_dropout_experiments.py first."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = pd.read_csv(INPUT_CSV)

    required_columns = {
        "scenario",
        "y_true",
        "y_prob_mean",
    }

    missing_columns = required_columns - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            f"Prediction file is missing columns: {sorted(missing_columns)}"
        )

    if predictions.empty:
        raise ValueError("Prediction file contains no rows.")

    actual_scenarios = set(predictions["scenario"].unique())

    if actual_scenarios != set(SCENARIOS):
        raise ValueError(
            "Scenario mismatch.\n"
            f"Expected: {SCENARIOS}\n"
            f"Found: {sorted(actual_scenarios)}"
        )

    summary_records = []
    bins_records = []
    plot_data = {}

    print("=" * 78)
    print("SensorGuard - Probability Calibration Evaluation")
    print("=" * 78)
    print(f"Subset: {SUBSET}")
    print(f"Input: {INPUT_CSV}")
    print(f"Equal-width bins: {N_BINS}")

    for scenario in SCENARIOS:
        frame = predictions.loc[
            predictions["scenario"] == scenario
        ].copy()

        y_true = frame["y_true"].to_numpy()
        y_prob = frame["y_prob_mean"].to_numpy(dtype=float)

        result = calibration_metrics(
            y_true=y_true,
            y_prob=y_prob,
            n_bins=N_BINS,
        )

        summary_records.append(
            {
                "subset": SUBSET,
                "scenario": scenario,
                "samples": result["samples"],
                "ece": result["ece"],
                "mce": result["mce"],
                "brier": result["brier"],
                "log_loss": result["log_loss"],
                "n_bins": result["n_bins"],
                "nonempty_bins": result["nonempty_bins"],
            }
        )

        for bin_result in result["calibration_bins"]:
            bins_records.append(
                {
                    "subset": SUBSET,
                    "scenario": scenario,
                    **bin_result,
                }
            )

        plot_data[scenario] = result["calibration_bins"]

        print(
            f"{scenario:20s} "
            f"ECE={result['ece']:.5f} | "
            f"MCE={result['mce']:.5f} | "
            f"Brier={result['brier']:.5f} | "
            f"LogLoss={result['log_loss']:.5f}"
        )

    summary_df = pd.DataFrame(summary_records)
    bins_df = pd.DataFrame(bins_records)

    summary_csv = OUTPUT_DIR / "calibration_metrics.csv"
    summary_json = OUTPUT_DIR / "calibration_metrics.json"
    bins_csv = OUTPUT_DIR / "calibration_bins.csv"
    reliability_png = OUTPUT_DIR / "reliability_diagrams.png"

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    bins_df.to_csv(
        bins_csv,
        index=False,
    )

    report = {
        "project": "SensorGuard",
        "experiment": "probability_calibration",
        "subset": SUBSET,
        "input_file": str(INPUT_CSV),
        "n_bins": N_BINS,
        "binning_strategy": "equal_width",
        "calibrator_fitted": False,
        "probability_source": (
            "Mean probability across 50 MC-Dropout passes"
        ),
        "scenarios": summary_records,
        "interpretation": (
            "ECE and MCE summarize bin-level calibration gaps. "
            "Brier score and log loss measure probabilistic prediction "
            "quality. These metrics evaluate the saved predictions; "
            "they do not establish that a calibration method improves "
            "performance on unseen data."
        ),
    }

    with summary_json.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            allow_nan=False,
        )

    # One plot with all six scenarios.
    fig, ax = plt.subplots(figsize=(9, 8))

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    for scenario in SCENARIOS:
        rows = plot_data[scenario]

        if not rows:
            continue

        confidence = [
            item["mean_confidence"] for item in rows
        ]

        observed = [
            item["observed_frequency"] for item in rows
        ]

        counts = [
            item["count"] for item in rows
        ]

        # Marker size indicates the number of samples in each bin.
        sizes = [
            20 + 100 * count / max(counts)
            for count in counts
        ]

        ax.plot(
            confidence,
            observed,
            marker="o",
            markersize=4,
            label=scenario,
        )

        ax.scatter(
            confidence,
            observed,
            s=sizes,
            alpha=0.35,
        )

    ax.set_xlabel("Mean predicted fault probability")
    ax.set_ylabel("Observed fault frequency")
    ax.set_title(
        f"SensorGuard Reliability Diagram - {SUBSET}"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    fig.savefig(
        reliability_png,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nCalibration comparison:")
    print(
        summary_df[
            ["scenario", "ece", "mce", "brier", "log_loss"]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    print("\nSaved files:")
    print(f"  Metrics:          {summary_csv}")
    print(f"  JSON report:      {summary_json}")
    print(f"  Calibration bins: {bins_csv}")
    print(f"  Reliability plot: {reliability_png}")

    print("\nCalibration evaluation completed successfully.")


if __name__ == "__main__":
    main()
