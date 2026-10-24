
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "FD001"
    / "temperature_scaling"
    / "temperature_scaling_predictions.csv"
)

BIN_COUNTS = (10, 15, 20)


def find_column(df, candidates, description):
    """Find a matching column name, ignoring capitalization."""
    lookup = {column.lower(): column for column in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Could not identify the {description} column.\n"
        f"Available columns: {list(df.columns)}\n"
        f"Update the candidate list in this script."
    )


def calculate_ece_and_audit(y_true, probabilities, n_bins):
    """Calculate ECE and print each bin's contribution independently."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)

    ece = 0.0
    signed_gap_sum = 0.0
    occupied = 0
    signs = []

    print(f"\n  BIN-BY-BIN AUDIT: {n_bins} bins")
    print(
        "  Bin | Count | Mean probability | Positive rate "
        "| Signed gap | Weighted contribution"
    )
    print("  " + "-" * 100)

    for i in range(n_bins):
        lower = edges[i]
        upper = edges[i + 1]

        # Left-inclusive, right-exclusive bins, except the final bin.
        if i == n_bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)

        count = int(mask.sum())

        if count == 0:
            print(
                f"  {i + 1:>3} |     0 |"
                f" {lower:.3f} to {upper:.3f} | EMPTY"
            )
            continue

        mean_probability = float(probabilities[mask].mean())
        observed_rate = float(y_true[mask].mean())

        # Positive means predicted probability exceeds observed frequency.
        signed_gap = mean_probability - observed_rate
        weighted_contribution = (
            count / total * abs(signed_gap)
        )

        ece += weighted_contribution
        signed_gap_sum += count / total * signed_gap
        occupied += 1

        if not np.isclose(signed_gap, 0.0, atol=1e-12):
            signs.append(np.sign(signed_gap))

        print(
            f"  {i + 1:>3} | {count:>5} |"
            f" {mean_probability:>16.8f} |"
            f" {observed_rate:>12.8f} |"
            f" {signed_gap:>11.8f} |"
            f" {weighted_contribution:>21.10f}"
        )

    prevalence = float(y_true.mean())
    mean_probability = float(probabilities.mean())
    absolute_overall_gap = abs(mean_probability - prevalence)

    print("  " + "-" * 100)
    print(f"  Samples:                     {total}")
    print(f"  Occupied bins:               {occupied}/{n_bins}")
    print(f"  Mean predicted probability:  {mean_probability:.10f}")
    print(f"  Observed positive rate:      {prevalence:.10f}")
    print(f"  Overall signed gap:          {signed_gap_sum:.10f}")
    print(f"  Absolute overall gap:        {absolute_overall_gap:.10f}")
    print(f"  Independently summed ECE:    {ece:.10f}")

    if len(signs) == 0:
        print("  Gap direction:               All gaps are approximately zero.")
    elif all(sign > 0 for sign in signs):
        print("  Gap direction:               ALL nonzero gaps are positive.")
        print(
            "  Interpretation: predictions exceed observed rates in every "
            "nonzero-gap bin; ECE may be bin-invariant."
        )
    elif all(sign < 0 for sign in signs):
        print("  Gap direction:               ALL nonzero gaps are negative.")
        print(
            "  Interpretation: predictions fall below observed rates in "
            "every nonzero-gap bin; ECE may be bin-invariant."
        )
    else:
        print("  Gap direction:               MIXED positive and negative gaps.")
        print(
            "  Interpretation: exact bin-invariance is less easily explained "
            "by a common gap direction; inspect contributions carefully."
        )

    return ece, occupied


def main():
    print("SensorGuard ECE contribution audit")
    print("=" * 100)
    print(f"Reading: {PREDICTIONS_FILE}")

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    print("\nAvailable CSV columns:")
    print(list(df.columns))

    scenario_col = find_column(
        df, ["scenario"], "scenario"
    )
    y_col = find_column(
        df, ["y_true", "true_label", "label"], "true-label"
    )

    uncalibrated_col = find_column(
        df,
        [
            "probability_uncalibrated",
            "y_prob_uncalibrated",
            "y_prob_mean",
            "prob_uncalibrated",
            "y_prob",
        ],
        "uncalibrated probability",
    )

    calibrated_col = find_column(
        df,
        [
            "probability_temperature_scaled",
            "y_prob_temperature_scaled",
            "prob_temperature_scaled",
            "probability_scaled",
            "y_prob_scaled",
        ],
        "temperature-scaled probability",
    )

    print(f"\nScenario column:             {scenario_col}")
    print(f"True-label column:           {y_col}")
    print(f"Uncalibrated probability:    {uncalibrated_col}")
    print(f"Temperature-scaled prob.:    {calibrated_col}")

    for scenario in sorted(df[scenario_col].dropna().unique()):
        subset = df.loc[df[scenario_col] == scenario]

        print("\n" + "=" * 100)
        print(f"SCENARIO: {scenario} | rows={len(subset)}")
        print("=" * 100)

        y_raw = subset[y_col].to_numpy()
        if not np.isin(y_raw, [0, 1]).all():
            raise ValueError(f"Non-binary labels found in {scenario}")

        y_true = y_raw.astype(int)

        for method, column in [
            ("UNCALIBRATED", uncalibrated_col),
            ("TEMPERATURE-SCALED", calibrated_col),
        ]:
            probabilities = subset[column].to_numpy(dtype=float)

            if not np.isfinite(probabilities).all():
                raise ValueError(
                    f"Non-finite probabilities in {scenario}, {method}"
                )

            if ((probabilities < 0) | (probabilities > 1)).any():
                raise ValueError(
                    f"Probabilities outside [0, 1] in {scenario}, {method}"
                )

            print(f"\nMETHOD: {method}")

            for n_bins in BIN_COUNTS:
                calculate_ece_and_audit(
                    y_true, probabilities, n_bins
                )


if __name__ == "__main__":
    main()
