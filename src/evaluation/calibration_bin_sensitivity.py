
"""Compare ECE across different bin counts using existing predictions.

This is an evaluation-only analysis. It does not fit a calibrator
or modify the existing model, checkpoint, or predictions.
"""

from pathlib import Path

import pandas as pd

from src.uncertainty.calibration import expected_calibration_error


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "FD001"
    / "temperature_scaling"
    / "temperature_scaling_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "FD001"
    / "calibration_bin_sensitivity"
)

BIN_COUNTS = [10, 15, 20]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "scenario",
        "y_true",
        "probability_uncalibrated",
        "probability_temperature_scaled",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {sorted(missing)}"
        )

    records = []

    for scenario, group in df.groupby("scenario", sort=False):
        y_true = group["y_true"].to_numpy(dtype=int)

        for method, column in [
            ("uncalibrated", "probability_uncalibrated"),
            (
                "temperature_scaled",
                "probability_temperature_scaled",
            ),
        ]:
            probabilities = group[column].to_numpy(dtype=float)

            for n_bins in BIN_COUNTS:
                ece = expected_calibration_error(
                    y_true,
                    probabilities,
                    n_bins=n_bins,
                )

                records.append({
                    "scenario": scenario,
                    "method": method,
                    "n_bins": n_bins,
                    "samples": len(group),
                    "ece": float(ece),
                })

    results = pd.DataFrame(records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_csv = OUTPUT_DIR / "ece_bin_sensitivity.csv"
    results.to_csv(output_csv, index=False)

    comparison = results.pivot(
        index=["scenario", "n_bins"],
        columns="method",
        values="ece",
    ).reset_index()

    comparison["ece_change_calibrated_minus_raw"] = (
        comparison["temperature_scaled"]
        - comparison["uncalibrated"]
    )

    comparison["lower_ece_method"] = comparison.apply(
        lambda row: (
            "temperature_scaled"
            if row["temperature_scaled"] < row["uncalibrated"]
            else (
                "uncalibrated"
                if row["uncalibrated"] < row["temperature_scaled"]
                else "tie"
            )
        ),
        axis=1,
    )

    comparison_file = OUTPUT_DIR / "ece_bin_sensitivity_comparison.csv"
    comparison.to_csv(comparison_file, index=False)

    print("\nECE by scenario, method, and bin count:")
    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nComparison:")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print(f"\nSaved: {output_csv}")
    print(f"Saved: {comparison_file}")


if __name__ == "__main__":
    main()
