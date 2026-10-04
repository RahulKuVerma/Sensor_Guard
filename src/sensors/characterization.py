from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.data.loader import load_dataset
from src.sensors.analyzer import (
    SENSOR_COLUMNS,
    generate_sensor_audit,
)
from src.sensors.operating_conditions import (
    generate_operating_condition_audit,
)
from src.sensors.redundancy import (
    generate_redundancy_audit,
)


# ============================================================
# SensorGuard - Consolidated Sensor Characterization
# ============================================================

SUBSETS = [
    "FD001",
    "FD002",
    "FD003",
    "FD004",
]

RESULTS_DIR = Path("results") / "metrics"

REDUNDANCY_THRESHOLD = 0.90


def build_sensor_characterization(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine all previously generated sensor-audit information
    into one consolidated characterization table.
    """

    # --------------------------------------------------------
    # Basic sensor audit
    # --------------------------------------------------------

    sensor_audit = generate_sensor_audit(df)

    statistics = sensor_audit["statistics"].copy()
    cycle_correlation = sensor_audit[
        "cycle_correlation"
    ].copy()
    rul_correlation = sensor_audit[
        "rul_correlation"
    ].copy()
    engine_variation = sensor_audit[
        "engine_variation"
    ].copy()
    ranges = sensor_audit["ranges"].copy()

    # --------------------------------------------------------
    # Operating-condition audit
    # --------------------------------------------------------

    operating_audit = (
        generate_operating_condition_audit(df)
    )

    setting_sensitivity = operating_audit[
        "sensor_setting_sensitivity"
    ].copy()

    # --------------------------------------------------------
    # Redundancy audit
    # --------------------------------------------------------

    redundancy_audit = generate_redundancy_audit(
        df,
        threshold=REDUNDANCY_THRESHOLD,
    )

    redundancy_scores = redundancy_audit[
        "redundancy_scores"
    ].copy()

    redundancy_counts = redundancy_audit[
        "redundancy_counts"
    ].copy()

    # --------------------------------------------------------
    # Rename columns before merging
    # --------------------------------------------------------

    statistics = statistics.rename(
        columns={
            "std": "standard_deviation",
            "variance": "variance",
        }
    )

    cycle_correlation = cycle_correlation.rename(
        columns={
            "cycle_correlation": "cycle_correlation",
            "absolute_cycle_correlation":
                "absolute_cycle_correlation",
        }
    )

    rul_correlation = rul_correlation.rename(
        columns={
            "rul_correlation": "rul_correlation",
            "absolute_rul_correlation":
                "absolute_rul_correlation",
        }
    )

    engine_variation = engine_variation.rename(
        columns={
            "engine_mean_std":
                "engine_mean_standard_deviation",
        }
    )

    setting_sensitivity = setting_sensitivity.rename(
        columns={
            "max_setting_correlation":
                "max_setting_correlation",
        }
    )

    # --------------------------------------------------------
    # Merge all audit tables
    # --------------------------------------------------------

    characterization = statistics.copy()

    characterization = characterization.merge(
        cycle_correlation,
        on="sensor",
        how="left",
    )

    characterization = characterization.merge(
        rul_correlation,
        on="sensor",
        how="left",
    )

    characterization = characterization.merge(
        engine_variation,
        on="sensor",
        how="left",
    )

    characterization = characterization.merge(
        ranges,
        on="sensor",
        how="left",
    )

    characterization = characterization.merge(
        setting_sensitivity,
        on="sensor",
        how="left",
    )

    characterization = characterization.merge(
        redundancy_scores,
        on="sensor",
        how="left",
    )

    characterization = characterization.merge(
        redundancy_counts,
        on="sensor",
        how="left",
    )

    # --------------------------------------------------------
    # Add useful characterization flags
    # --------------------------------------------------------

    characterization["is_constant"] = (
        characterization["variance"]
        .fillna(0)
        .eq(0)
    )

    characterization["has_variation"] = (
        ~characterization["is_constant"]
    )

    characterization["strong_rul_relationship"] = (
        characterization[
            "absolute_rul_correlation"
        ]
        >= 0.50
    )

    characterization[
        "strong_cycle_relationship"
    ] = (
        characterization[
            "absolute_cycle_correlation"
        ]
        >= 0.50
    )

    characterization[
        "high_operating_condition_sensitivity"
    ] = (
        characterization[
            "max_setting_correlation"
        ]
        >= 0.50
    )

    characterization[
        "high_sensor_redundancy"
    ] = (
        characterization[
            "mean_absolute_correlation"
        ]
        >= 0.80
    )

    characterization[
        "has_high_correlation_partner"
    ] = (
        characterization[
            "high_correlation_count"
        ]
        > 0
    )

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    preferred_columns = [
        "sensor",
        "mean",
        "standard_deviation",
        "variance",
        "min",
        "max",
        "range",
        "unique_values",
        "missing_values",
        "is_constant",
        "has_variation",
        "cycle_correlation",
        "absolute_cycle_correlation",
        "rul_correlation",
        "absolute_rul_correlation",
        "strong_rul_relationship",
        "strong_cycle_relationship",
        "engine_mean_standard_deviation",
        "engine_mean_min",
        "engine_mean_max",
        "max_setting_correlation",
        "most_sensitive_setting",
        "high_operating_condition_sensitivity",
        "mean_absolute_correlation",
        "maximum_absolute_correlation",
        "high_correlation_count",
        "high_sensor_redundancy",
        "has_high_correlation_partner",
    ]

    existing_columns = [
        column
        for column in preferred_columns
        if column in characterization.columns
    ]

    remaining_columns = [
        column
        for column in characterization.columns
        if column not in existing_columns
    ]

    characterization = characterization[
        existing_columns + remaining_columns
    ]

    # --------------------------------------------------------
    # Sort by sensor number
    # --------------------------------------------------------

    characterization["_sensor_number"] = (
        characterization["sensor"]
        .str.extract(r"(\d+)")
        .astype(int)
    )

    characterization = (
        characterization
        .sort_values("_sensor_number")
        .drop(columns="_sensor_number")
        .reset_index(drop=True)
    )

    return characterization


def print_characterization_summary(
    characterization: pd.DataFrame,
    subset: str,
):
    """
    Print a concise summary of the consolidated
    sensor characterization.
    """

    print("\n" + "=" * 100)
    print(
        f"SensorGuard - Sensor Characterization Summary "
        f"- {subset}"
    )
    print("=" * 100)

    print("\nComplete Characterization Table")
    print("-" * 100)

    display_columns = [
        "sensor",
        "variance",
        "range",
        "absolute_rul_correlation",
        "max_setting_correlation",
        "mean_absolute_correlation",
        "high_correlation_count",
        "is_constant",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in characterization.columns
    ]

    print(
        characterization[
            available_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # Constant sensors
    # --------------------------------------------------------

    constant_sensors = characterization.loc[
        characterization["is_constant"],
        "sensor",
    ].tolist()

    print("\nConstant Sensors")
    print("-" * 100)

    if constant_sensors:
        print(", ".join(constant_sensors))
    else:
        print("None")

    # --------------------------------------------------------
    # Strong RUL relationship
    # --------------------------------------------------------

    rul_sensors = characterization.loc[
        characterization["strong_rul_relationship"],
        "sensor",
    ].tolist()

    print("\nSensors With |RUL Correlation| >= 0.50")
    print("-" * 100)

    if rul_sensors:
        print(", ".join(rul_sensors))
    else:
        print("None")

    # --------------------------------------------------------
    # Operating condition sensitivity
    # --------------------------------------------------------

    condition_sensors = characterization.loc[
        characterization[
            "high_operating_condition_sensitivity"
        ],
        "sensor",
    ].tolist()

    print(
        "\nSensors With Operating-Condition "
        "|Correlation| >= 0.50"
    )
    print("-" * 100)

    if condition_sensors:
        print(", ".join(condition_sensors))
    else:
        print("None")

    # --------------------------------------------------------
    # High redundancy
    # --------------------------------------------------------

    redundant_sensors = characterization.loc[
        characterization[
            "high_sensor_redundancy"
        ],
        "sensor",
    ].tolist()

    print(
        "\nSensors With Mean Absolute "
        "Correlation >= 0.80"
    )
    print("-" * 100)

    if redundant_sensors:
        print(", ".join(redundant_sensors))
    else:
        print("None")

    print("\n" + "=" * 100)


def save_characterization(
    characterization: pd.DataFrame,
    subset: str,
):
    """
    Save the consolidated characterization table.
    """

    subset_dir = RESULTS_DIR / subset

    subset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        subset_dir
        / "sensor_characterization.csv"
    )

    characterization.to_csv(
        output_file,
        index=False,
    )

    print(f"Saved: {output_file}")


def run_characterization():
    """
    Run sensor characterization for all C-MAPSS subsets.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "SensorGuard - Complete Sensor Characterization"
    )
    print("=" * 100)

    for subset in SUBSETS:

        print("\n" + "#" * 100)
        print(f"Processing {subset}")
        print("#" * 100)

        dataset = load_dataset(subset)

        train_df = dataset["train"]

        characterization = (
            build_sensor_characterization(
                train_df
            )
        )

        print_characterization_summary(
            characterization,
            subset,
        )

        save_characterization(
            characterization,
            subset,
        )

    print("\n" + "=" * 100)
    print(
        "Complete sensor characterization finished."
    )
    print("=" * 100)


if __name__ == "__main__":
    run_characterization()