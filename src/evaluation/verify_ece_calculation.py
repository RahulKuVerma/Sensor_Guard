
"""Independently verify equal-width ECE calculations."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "results"
    / "metrics"
    / "FD001"
    / "temperature_scaling"
    / "temperature_scaling_predictions.csv"
)


def independent_ece(y_true, probabilities, n_bins):
    """Compute ECE directly from equal-width bin masks."""

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(
        probabilities, dtype=float
    ).reshape(-1)

    if len(y_true) != len(probabilities) or len(y_true) == 0:
        raise ValueError("Inputs must be nonempty and equally sized.")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece = 0.0
    occupied_bins = 0

    for index in range(n_bins):
        lower = edges[index]
        upper = edges[index + 1]

        # Left-closed bins; the final bin also includes probability 1.
        if index == n_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        occupied_bins += 1

        confidence = float(probabilities[mask].mean())
        observed_frequency = float(y_true[mask].mean())

        ece += (
            count / total
        ) * abs(confidence - observed_frequency)

    return float(ece), occupied_bins


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print("Independent ECE verification")
    print("=" * 90)

    for scenario, group in df.groupby("scenario", sort=False):
        y = group["y_true"].to_numpy(dtype=int)

        for method, column in [
            ("uncalibrated", "probability_uncalibrated"),
            (
                "temperature_scaled",
                "probability_temperature_scaled",
            ),
        ]:
            p = group[column].to_numpy(dtype=float)

            for n_bins in [10, 15, 20]:
                ece, occupied = independent_ece(
                    y, p, n_bins
                )

                print(
                    f"{scenario:18s} | "
                    f"{method:22s} | "
                    f"bins={n_bins:2d} | "
                    f"occupied={occupied:2d} | "
                    f"ECE={ece:.10f}"
                )


if __name__ == "__main__":
    main()
