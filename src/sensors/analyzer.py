from typing import Dict

import numpy as np
import pandas as pd


# ============================================================
# SensorGuard - Sensor Analysis
# ============================================================


SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
]


def analyze_sensor_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate basic statistical properties of every sensor.
    """

    records = []

    for sensor in SENSOR_COLUMNS:

        if sensor not in df.columns:
            continue

        series = df[sensor]

        records.append(
            {
                "sensor": sensor,
                "mean": series.mean(),
                "std": series.std(),
                "variance": series.var(),
                "min": series.min(),
                "max": series.max(),
                "unique_values": series.nunique(),
                "missing_values": series.isna().sum(),
            }
        )

    return pd.DataFrame(records)


def analyze_sensor_cycle_correlation(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Pearson correlation between each sensor
    and engine cycle.
    """

    records = []

    for sensor in SENSOR_COLUMNS:

        if sensor not in df.columns:
            continue

        correlation = df[sensor].corr(df["cycle"])

        records.append(
            {
                "sensor": sensor,
                "cycle_correlation": correlation,
                "absolute_cycle_correlation": abs(correlation),
            }
        )

    return pd.DataFrame(records)


def analyze_sensor_engine_variation(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Measure how much each sensor varies across engines.
    """

    records = []

    for sensor in SENSOR_COLUMNS:

        if sensor not in df.columns:
            continue

        engine_means = (
            df.groupby("engine_id")[sensor]
            .mean()
        )

        records.append(
            {
                "sensor": sensor,
                "engine_mean_std": engine_means.std(),
                "engine_mean_min": engine_means.min(),
                "engine_mean_max": engine_means.max(),
            }
        )

    return pd.DataFrame(records)


def analyze_sensor_ranges(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the dynamic range of every sensor.
    """

    records = []

    for sensor in SENSOR_COLUMNS:

        if sensor not in df.columns:
            continue

        sensor_min = df[sensor].min()
        sensor_max = df[sensor].max()

        records.append(
            {
                "sensor": sensor,
                "range": sensor_max - sensor_min,
            }
        )

    return pd.DataFrame(records)


def generate_sensor_audit(
    df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Run the complete sensor audit.
    """

    statistics = analyze_sensor_statistics(df)

    cycle_correlation = analyze_sensor_cycle_correlation(df)

    engine_variation = analyze_sensor_engine_variation(df)

    ranges = analyze_sensor_ranges(df)

    audit = {
        "statistics": statistics,
        "cycle_correlation": cycle_correlation,
        "engine_variation": engine_variation,
        "ranges": ranges,
    }

    return audit


def print_sensor_audit(
    audit: Dict[str, pd.DataFrame],
):
    """
    Print the sensor audit in a readable format.
    """

    statistics = audit["statistics"]
    cycle_correlation = audit["cycle_correlation"]
    engine_variation = audit["engine_variation"]
    ranges = audit["ranges"]

    combined = (
        statistics
        .merge(
            cycle_correlation,
            on="sensor",
            how="left",
        )
        .merge(
            engine_variation,
            on="sensor",
            how="left",
        )
        .merge(
            ranges,
            on="sensor",
            how="left",
        )
    )

    print("\n" + "=" * 100)
    print("SensorGuard - Sensor Audit")
    print("=" * 100)

    print("\nSensor Statistics")
    print("-" * 100)

    print(
        combined[
            [
                "sensor",
                "mean",
                "std",
                "variance",
                "unique_values",
                "min",
                "max",
            ]
        ].to_string(index=False)
    )

    print("\nCycle Correlation")
    print("-" * 100)

    cycle_table = (
        combined[
            [
                "sensor",
                "cycle_correlation",
                "absolute_cycle_correlation",
            ]
        ]
        .sort_values(
            "absolute_cycle_correlation",
            ascending=False,
        )
    )

    print(
        cycle_table.to_string(index=False)
    )

    print("\nEngine-Level Variation")
    print("-" * 100)

    engine_table = (
        combined[
            [
                "sensor",
                "engine_mean_std",
                "engine_mean_min",
                "engine_mean_max",
            ]
        ]
        .sort_values(
            "engine_mean_std",
            ascending=False,
        )
    )

    print(
        engine_table.to_string(index=False)
    )

    print("\nDynamic Range")
    print("-" * 100)

    range_table = (
        combined[
            [
                "sensor",
                "range",
            ]
        ]
        .sort_values(
            "range",
            ascending=False,
        )
    )

    print(
        range_table.to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("Sensor audit completed.")
    print("=" * 100)