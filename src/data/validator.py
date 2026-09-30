from typing import Dict, List

import pandas as pd


# ============================================================
# SensorGuard - C-MAPSS Dataset Validator
# ============================================================

EXPECTED_COLUMN_COUNT = 26

ID_COLUMNS = [
    "engine_id",
    "cycle",
]

SETTING_COLUMNS = [
    "setting_1",
    "setting_2",
    "setting_3",
]

SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
]

EXPECTED_COLUMNS = (
    ID_COLUMNS
    + SETTING_COLUMNS
    + SENSOR_COLUMNS
)


def validate_columns(df: pd.DataFrame) -> List[str]:
    """
    Validate the C-MAPSS dataframe columns.
    """

    errors = []

    if len(df.columns) != EXPECTED_COLUMN_COUNT:
        errors.append(
            f"Expected {EXPECTED_COLUMN_COUNT} columns, "
            f"found {len(df.columns)}."
        )

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    unexpected_columns = [
        column
        for column in df.columns
        if column not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        errors.append(
            f"Missing columns: {missing_columns}"
        )

    if unexpected_columns:
        errors.append(
            f"Unexpected columns: {unexpected_columns}"
        )

    return errors


def validate_missing_values(df: pd.DataFrame) -> Dict:
    """
    Check for missing values in the dataframe.
    """

    missing_counts = df.isna().sum()

    total_missing = int(missing_counts.sum())

    columns_with_missing = {
        column: int(count)
        for column, count in missing_counts.items()
        if count > 0
    }

    return {
        "total_missing": total_missing,
        "columns_with_missing": columns_with_missing,
        "has_missing": total_missing > 0,
    }


def validate_duplicates(df: pd.DataFrame) -> Dict:
    """
    Check for duplicate rows.
    """

    duplicate_count = int(df.duplicated().sum())

    return {
        "duplicate_rows": duplicate_count,
        "has_duplicates": duplicate_count > 0,
    }


def validate_numeric_columns(df: pd.DataFrame) -> Dict:
    """
    Check whether all expected numerical columns
    contain numeric values.
    """

    non_numeric_columns = []

    for column in EXPECTED_COLUMNS:
        if column not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[column]):
            non_numeric_columns.append(column)

    return {
        "non_numeric_columns": non_numeric_columns,
        "all_numeric": len(non_numeric_columns) == 0,
    }


def validate_engine_ids(df: pd.DataFrame) -> Dict:
    """
    Validate engine IDs.
    """

    if "engine_id" not in df.columns:
        return {
            "valid": False,
            "reason": "engine_id column not found.",
        }

    engine_ids = df["engine_id"]

    return {
        "valid": (
            engine_ids.notna().all()
            and (engine_ids > 0).all()
            and pd.api.types.is_numeric_dtype(engine_ids)
        ),
        "engine_count": int(engine_ids.nunique()),
        "min_engine_id": int(engine_ids.min()),
        "max_engine_id": int(engine_ids.max()),
    }


def validate_cycle_order(df: pd.DataFrame) -> Dict:
    """
    Check that cycles increase correctly within each engine.
    """

    if "engine_id" not in df.columns or "cycle" not in df.columns:
        return {
            "valid": False,
            "invalid_engines": [],
            "reason": "engine_id or cycle column missing.",
        }

    invalid_engines = []

    grouped = df.groupby("engine_id", sort=False)

    for engine_id, engine_df in grouped:
        cycles = engine_df["cycle"].to_numpy()

        if len(cycles) == 0:
            invalid_engines.append(engine_id)
            continue

        if cycles[0] != 1:
            invalid_engines.append(engine_id)
            continue

        if not all(cycles[i] < cycles[i + 1] for i in range(len(cycles) - 1)):
            invalid_engines.append(engine_id)

    return {
        "valid": len(invalid_engines) == 0,
        "invalid_engines": invalid_engines,
    }


def validate_sensor_columns(df: pd.DataFrame) -> Dict:
    """
    Validate the number of sensor columns.
    """

    available_sensors = [
        column
        for column in df.columns
        if column.startswith("sensor_")
    ]

    return {
        "sensor_count": len(available_sensors),
        "expected_sensor_count": len(SENSOR_COLUMNS),
        "valid": len(available_sensors) == len(SENSOR_COLUMNS),
        "sensor_columns": available_sensors,
    }


def validate_dataframe(df: pd.DataFrame) -> Dict:
    """
    Perform all validation checks on one dataframe.
    """

    column_errors = validate_columns(df)
    missing_info = validate_missing_values(df)
    duplicate_info = validate_duplicates(df)
    numeric_info = validate_numeric_columns(df)
    engine_info = validate_engine_ids(df)
    cycle_info = validate_cycle_order(df)
    sensor_info = validate_sensor_columns(df)

    valid = (
        len(column_errors) == 0
        and not missing_info["has_missing"]
        and not duplicate_info["has_duplicates"]
        and numeric_info["all_numeric"]
        and engine_info["valid"]
        and cycle_info["valid"]
        and sensor_info["valid"]
    )

    return {
        "valid": valid,
        "rows": len(df),
        "columns": len(df.columns),
        "column_errors": column_errors,
        "missing": missing_info,
        "duplicates": duplicate_info,
        "numeric": numeric_info,
        "engines": engine_info,
        "cycles": cycle_info,
        "sensors": sensor_info,
    }


def validate_rul(rul_df: pd.DataFrame) -> Dict:
    """
    Validate a C-MAPSS RUL dataframe.
    """

    errors = []

    if "rul" not in rul_df.columns:
        errors.append("RUL column is missing.")

    else:
        if rul_df["rul"].isna().any():
            errors.append("RUL contains missing values.")

        if not pd.api.types.is_numeric_dtype(rul_df["rul"]):
            errors.append("RUL column is not numeric.")

        if (rul_df["rul"] < 0).any():
            errors.append("RUL contains negative values.")

    return {
        "valid": len(errors) == 0,
        "rows": len(rul_df),
        "errors": errors,
    }


def print_validation_report(
    subset: str,
    train_result: Dict,
    test_result: Dict,
    rul_result: Dict,
):
    """
    Print a readable validation report.
    """

    print("\n" + "=" * 70)
    print(f"SensorGuard - Validation Report: {subset}")
    print("=" * 70)

    print("\nTRAIN DATA")
    print("-" * 70)

    print(f"Valid              : {train_result['valid']}")
    print(f"Rows               : {train_result['rows']}")
    print(f"Columns            : {train_result['columns']}")
    print(f"Missing values     : {train_result['missing']['total_missing']}")
    print(f"Duplicate rows     : {train_result['duplicates']['duplicate_rows']}")
    print(f"Engine count       : {train_result['engines']['engine_count']}")
    print(f"Sensor count       : {train_result['sensors']['sensor_count']}")
    print(f"Cycle order valid  : {train_result['cycles']['valid']}")

    print("\nTEST DATA")
    print("-" * 70)

    print(f"Valid              : {test_result['valid']}")
    print(f"Rows               : {test_result['rows']}")
    print(f"Columns            : {test_result['columns']}")
    print(f"Missing values     : {test_result['missing']['total_missing']}")
    print(f"Duplicate rows     : {test_result['duplicates']['duplicate_rows']}")
    print(f"Engine count       : {test_result['engines']['engine_count']}")
    print(f"Sensor count       : {test_result['sensors']['sensor_count']}")
    print(f"Cycle order valid  : {test_result['cycles']['valid']}")

    print("\nRUL DATA")
    print("-" * 70)

    print(f"Valid              : {rul_result['valid']}")
    print(f"Rows               : {rul_result['rows']}")

    if rul_result["errors"]:
        print(f"Errors             : {rul_result['errors']}")

    print("\n" + "=" * 70)

    overall_valid = (
        train_result["valid"]
        and test_result["valid"]
        and rul_result["valid"]
    )

    if overall_valid:
        print("RESULT: VALID")
    else:
        print("RESULT: VALIDATION FAILED")

    print("=" * 70)


if __name__ == "__main__":

    from src.data.loader import load_dataset

    subset = "FD001"

    print("Loading dataset...")

    dataset = load_dataset(subset)

    train_df = dataset["train"]
    test_df = dataset["test"]
    rul_df = dataset["rul"]

    train_result = validate_dataframe(train_df)
    test_result = validate_dataframe(test_df)
    rul_result = validate_rul(rul_df)

    print_validation_report(
        subset,
        train_result,
        test_result,
        rul_result,
    )