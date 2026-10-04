from typing import List, Tuple

import numpy as np
import pandas as pd


SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
]

SETTING_COLUMNS = [
    "setting_1",
    "setting_2",
    "setting_3",
]


def sort_engine_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sort observations by engine and cycle.

    This guarantees chronological ordering before
    feature construction or window generation.
    """

    required_columns = {
        "engine_id",
        "cycle",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return (
        df.sort_values(
            ["engine_id", "cycle"]
        )
        .reset_index(drop=True)
    )


def add_training_rul(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add true training RUL.

    RUL at time t is:
        maximum engine cycle - current cycle

    This function is intended for training trajectories,
    where the failure/end-of-run cycle is known.
    """

    df = sort_engine_data(df).copy()

    max_cycle = (
        df.groupby("engine_id")["cycle"]
        .transform("max")
    )

    df["rul"] = (
        max_cycle - df["cycle"]
    )

    return df


def add_early_fault_target(
    df: pd.DataFrame,
    horizon: int = 30,
) -> pd.DataFrame:
    """
    Construct the early-fault binary target.

    y = 1  if RUL <= horizon
    y = 0  otherwise

    The target is constructed from RUL and does not use
    future sensor observations as input features.
    """

    if "rul" not in df.columns:
        raise ValueError(
            "RUL column not found. "
            "Run add_training_rul() first."
        )

    if horizon <= 0:
        raise ValueError(
            "horizon must be greater than zero."
        )

    result = df.copy()

    result["early_fault"] = (
        result["rul"] <= horizon
    ).astype(int)

    return result


def remove_constant_sensors(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove sensors having zero variance in the supplied
    dataframe.

    Returns:
        processed dataframe
        list of removed sensors

    This function is intentionally explicit rather than
    automatically applied to the complete SensorGuard pipeline.
    """

    available_sensors = [
        sensor
        for sensor in SENSOR_COLUMNS
        if sensor in df.columns
    ]

    constant_sensors = []

    for sensor in available_sensors:

        if df[sensor].nunique(dropna=False) <= 1:
            constant_sensors.append(sensor)

    processed = df.drop(
        columns=constant_sensors
    )

    return processed, constant_sensors


def select_sensor_columns(
    df: pd.DataFrame,
    sensors: List[str],
) -> pd.DataFrame:
    """
    Select a specified sensor subset.

    This allows SensorGuard experiments to explicitly define
    which channels are available to the model.
    """

    missing = [
        sensor
        for sensor in sensors
        if sensor not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Requested sensors are missing: {missing}"
        )

    return df[
        ["engine_id", "cycle"] + sensors
    ].copy()


def validate_rul_target(
    df: pd.DataFrame,
) -> dict:
    """
    Validate RUL and early-fault target construction.
    """

    result = {
        "valid": True,
        "rows": len(df),
        "missing_rul": 0,
        "negative_rul": 0,
        "missing_target": 0,
        "target_values": [],
    }

    if "rul" not in df.columns:
        result["valid"] = False
        result["error"] = "Missing RUL column."
        return result

    if "early_fault" not in df.columns:
        result["valid"] = False
        result["error"] = "Missing early_fault column."
        return result

    result["missing_rul"] = int(
        df["rul"].isna().sum()
    )

    result["negative_rul"] = int(
        (df["rul"] < 0).sum()
    )

    result["missing_target"] = int(
        df["early_fault"].isna().sum()
    )

    result["target_values"] = sorted(
        df["early_fault"]
        .dropna()
        .unique()
        .tolist()
    )

    if result["missing_rul"] > 0:
        result["valid"] = False

    if result["negative_rul"] > 0:
        result["valid"] = False

    if result["missing_target"] > 0:
        result["valid"] = False

    if not set(
        result["target_values"]
    ).issubset({0, 1}):
        result["valid"] = False

    return result


def create_training_dataset(
    df: pd.DataFrame,
    horizon: int = 30,
) -> pd.DataFrame:
    """
    Complete training-target preparation.

    Steps:
        1. Sort by engine/cycle
        2. Construct RUL
        3. Construct early-fault target
    """

    processed = sort_engine_data(df)

    processed = add_training_rul(
        processed
    )

    processed = add_early_fault_target(
        processed,
        horizon=horizon,
    )

    return processed


def summarize_target(
    df: pd.DataFrame,
) -> dict:
    """
    Generate class-distribution statistics.
    """

    if "early_fault" not in df.columns:
        raise ValueError(
            "early_fault column not found."
        )

    counts = (
        df["early_fault"]
        .value_counts()
        .sort_index()
    )

    total = len(df)

    negative_count = int(
        counts.get(0, 0)
    )

    positive_count = int(
        counts.get(1, 0)
    )

    return {
        "total_rows": total,
        "normal_rows": negative_count,
        "early_fault_rows": positive_count,
        "early_fault_ratio": (
            positive_count / total
            if total > 0
            else np.nan
        ),
    }