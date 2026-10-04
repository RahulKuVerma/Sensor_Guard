from typing import Dict

import numpy as np
import pandas as pd


# ============================================================
# SensorGuard - Sensor Redundancy Analysis
# ============================================================

SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
]


def calculate_sensor_correlation_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the Pearson correlation matrix between
    all available sensor channels.
    """

    available_sensors = [
        sensor
        for sensor in SENSOR_COLUMNS
        if sensor in df.columns
    ]

    sensor_data = df[available_sensors]

    correlation_matrix = sensor_data.corr()

    return correlation_matrix


def calculate_absolute_correlation_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the absolute Pearson correlation matrix.
    """

    correlation_matrix = (
        calculate_sensor_correlation_matrix(df)
    )

    return correlation_matrix.abs()


def find_highly_correlated_pairs(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.90,
) -> pd.DataFrame:
    """
    Find unique sensor pairs whose absolute correlation
    is greater than or equal to the specified threshold.

    The diagonal and duplicate pairs are excluded.
    """

    records = []

    sensors = correlation_matrix.columns.tolist()

    for i in range(len(sensors)):

        for j in range(i + 1, len(sensors)):

            sensor_a = sensors[i]
            sensor_b = sensors[j]

            correlation = correlation_matrix.loc[
                sensor_a,
                sensor_b,
            ]

            if pd.isna(correlation):
                continue

            absolute_correlation = abs(correlation)

            if absolute_correlation >= threshold:

                records.append(
                    {
                        "sensor_a": sensor_a,
                        "sensor_b": sensor_b,
                        "correlation": correlation,
                        "absolute_correlation": (
                            absolute_correlation
                        ),
                    }
                )

    if not records:
        return pd.DataFrame(
            columns=[
                "sensor_a",
                "sensor_b",
                "correlation",
                "absolute_correlation",
            ]
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            "absolute_correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def calculate_sensor_redundancy_scores(
    correlation_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the average absolute correlation of each sensor
    with all other sensors.

    Higher values indicate stronger overall redundancy.
    """

    records = []

    for sensor in correlation_matrix.columns:

        correlations = correlation_matrix[
            sensor
        ].drop(labels=[sensor])

        valid_correlations = correlations.dropna()

        if valid_correlations.empty:

            mean_absolute_correlation = np.nan
            maximum_absolute_correlation = np.nan

        else:

            absolute_correlations = (
                valid_correlations.abs()
            )

            mean_absolute_correlation = (
                absolute_correlations.mean()
            )

            maximum_absolute_correlation = (
                absolute_correlations.max()
            )

        records.append(
            {
                "sensor": sensor,
                "mean_absolute_correlation": (
                    mean_absolute_correlation
                ),
                "maximum_absolute_correlation": (
                    maximum_absolute_correlation
                ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            "mean_absolute_correlation",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )


def calculate_redundancy_counts(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.90,
) -> pd.DataFrame:
    """
    Count how many other sensors each sensor is highly
    correlated with.

    A high count indicates that the sensor belongs to a
    highly redundant sensor group.
    """

    records = []

    for sensor in correlation_matrix.columns:

        correlations = correlation_matrix[
            sensor
        ].drop(labels=[sensor])

        valid_correlations = correlations.dropna()

        high_correlation_count = int(
            (
                valid_correlations.abs()
                >= threshold
            ).sum()
        )

        records.append(
            {
                "sensor": sensor,
                "high_correlation_count": (
                    high_correlation_count
                ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            "high_correlation_count",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def generate_redundancy_audit(
    df: pd.DataFrame,
    threshold: float = 0.90,
) -> Dict[str, pd.DataFrame]:
    """
    Run the complete sensor redundancy analysis.
    """

    correlation_matrix = (
        calculate_sensor_correlation_matrix(df)
    )

    absolute_correlation_matrix = (
        correlation_matrix.abs()
    )

    highly_correlated_pairs = (
        find_highly_correlated_pairs(
            correlation_matrix,
            threshold=threshold,
        )
    )

    redundancy_scores = (
        calculate_sensor_redundancy_scores(
            correlation_matrix
        )
    )

    redundancy_counts = (
        calculate_redundancy_counts(
            correlation_matrix,
            threshold=threshold,
        )
    )

    return {
        "correlation_matrix": correlation_matrix,
        "absolute_correlation_matrix": (
            absolute_correlation_matrix
        ),
        "highly_correlated_pairs": (
            highly_correlated_pairs
        ),
        "redundancy_scores": redundancy_scores,
        "redundancy_counts": redundancy_counts,
    }


def print_redundancy_audit(
    audit: Dict[str, pd.DataFrame],
    threshold: float = 0.90,
):
    """
    Print the redundancy analysis in a readable format.
    """

    correlation_matrix = audit[
        "correlation_matrix"
    ]

    highly_correlated_pairs = audit[
        "highly_correlated_pairs"
    ]

    redundancy_scores = audit[
        "redundancy_scores"
    ]

    redundancy_counts = audit[
        "redundancy_counts"
    ]

    print("\n" + "=" * 100)
    print("SensorGuard - Sensor Redundancy Audit")
    print("=" * 100)

    # --------------------------------------------------------
    # Correlation Matrix
    # --------------------------------------------------------

    print("\nSensor Correlation Matrix")
    print("-" * 100)

    print(
        correlation_matrix.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # --------------------------------------------------------
    # Highly Correlated Pairs
    # --------------------------------------------------------

    print(
        f"\nHighly Correlated Sensor Pairs "
        f"(absolute correlation >= {threshold:.2f})"
    )

    print("-" * 100)

    if highly_correlated_pairs.empty:

        print(
            "No highly correlated sensor pairs found."
        )

    else:

        print(
            highly_correlated_pairs.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    # --------------------------------------------------------
    # Redundancy Scores
    # --------------------------------------------------------

    print("\nSensor Redundancy Scores")
    print("-" * 100)

    print(
        redundancy_scores.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # Redundancy Counts
    # --------------------------------------------------------

    print(
        f"\nHigh-Correlation Count "
        f"(absolute correlation >= {threshold:.2f})"
    )

    print("-" * 100)

    print(
        redundancy_counts.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("Sensor redundancy audit completed.")
    print("=" * 100)