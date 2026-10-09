
"""Diagnose ECE bin assignments and prediction variation."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.uncertainty.calibration import (
    calibration_bin_statistics,
    expected_calibration_error,
)


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "results"
    / "metrics"
    / "FD001"
    / "temperature_scaling"
    / "temperature_scaling_predictions.csv"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    probability_columns = [
        "probability_uncalibrated",
        "probability_temperature_scaled",
    ]

    required = {
        "scenario",
        "y_true",
        *probability_columns,
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    print("\n1. Probability variation by scenario")
    print("-" * 85)

    for scenario, group in df.groupby("scenario", sort=False):
        print(f"\nScenario: {scenario}")
        print(f"Rows: {len(group)}")
        print(f"Unique labels: {group['y_true'].nunique()}")

        for column in probability_columns:
            p = group[column].to_numpy(dtype=float)

            print(
                f"{column}: "
                f"min={p.min():.8f}, "
                f"max={p.max():.8f}, "
                f"unique={np.unique(p).size}, "
                f"mean={p.mean():.8f}, "
                f"std={p.std():.8f}"
            )

    print("\n\n2. Actual ECE and nonempty-bin counts")
    print("-" * 85)

    for scenario, group in df.groupby("scenario", sort=False):
        y = group["y_true"].to_numpy(dtype=int)

        for column in probability_columns:
            p = group[column].to_numpy(dtype=float)

            for n_bins in [10, 15, 20]:
                bins = calibration_bin_statistics(
                    y, p, n_bins=n_bins
                )

                ece = expected_calibration_error(
                    y, p, n_bins=n_bins
                )

                print(
                    f"{scenario:18s} "
                    f"{column:32s} "
                    f"bins={n_bins:2d} "
                    f"nonempty={len(bins):2d} "
                    f"ECE={ece:.10f}"
                )

                print(
                    "  counts:",
                    [item["count"] for item in bins],
                )

                print(
                    "  gaps:",
                    [
                        round(
                            item["absolute_calibration_error"],
                            6,
                        )
                        for item in bins
                    ],
                )


if __name__ == "__main__":
    main()
