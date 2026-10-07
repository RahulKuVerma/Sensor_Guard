from typing import Dict, List

import pandas as pd

from src.data.preprocessing import create_training_dataset
from src.data.scaler import SensorScaler
from src.data.test_targets import create_test_targets
from src.features.windows import (
    create_sliding_windows,
    split_engines,
)


DEFAULT_SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
]


def prepare_training_validation_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    sensor_columns: List[str] | None = None,
    horizon: int = 30,
    train_ratio: float = 0.85,
    validation_ratio: float = 0.15,
    window_size: int = 30,
    stride: int = 1,
    random_seed: int = 42,
) -> Dict[str, object]:

    if sensor_columns is None:
        sensor_columns = DEFAULT_SENSOR_COLUMNS.copy()

    # =========================================================
    # 1. Prepare training labels
    # =========================================================

    training = create_training_dataset(
        train_df,
        horizon=horizon,
    )

    # =========================================================
    # 2. Split ORIGINAL C-MAPSS TRAIN data
    #
    # 85% -> training
    # 15% -> validation
    #
    # Official C-MAPSS TEST remains completely separate.
    # =========================================================

    train_split, validation_split = split_engines(
        training,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        random_seed=random_seed,
    )

    # =========================================================
    # 3. Verify engine separation
    # =========================================================

    train_engines = set(
        train_split["engine_id"].unique()
    )

    validation_engines = set(
        validation_split["engine_id"].unique()
    )

    train_validation_overlap = (
        train_engines & validation_engines
    )

    if train_validation_overlap:
        raise RuntimeError(
            "Training/validation engine leakage detected: "
            f"{sorted(train_validation_overlap)}"
        )

    # =========================================================
    # 4. FIT SCALER ONLY ON TRAINING DATA
    # =========================================================

    scaler = SensorScaler(
        sensor_columns=sensor_columns
    )

    train_scaled = scaler.fit_transform(
        train_split
    )

    # =========================================================
    # 5. Transform validation using training scaler
    # =========================================================

    validation_scaled = scaler.transform(
        validation_split
    )

    # =========================================================
    # 6. Construct official C-MAPSS test targets
    # =========================================================

    official_test = create_test_targets(
        test_df=test_df,
        rul_df=rul_df,
        horizon=horizon,
    )

    # =========================================================
    # 7. Transform official test using SAME scaler
    # =========================================================

    test_scaled = scaler.transform(
        official_test
    )

    # =========================================================
    # 8. Generate training windows
    # =========================================================

    X_train, y_train, train_engine_ids = (
        create_sliding_windows(
            train_scaled,
            sensor_columns=sensor_columns,
            window_size=window_size,
            stride=stride,
        )
    )

    # =========================================================
    # 9. Generate validation windows
    # =========================================================

    (
        X_validation,
        y_validation,
        validation_engine_ids,
    ) = create_sliding_windows(
        validation_scaled,
        sensor_columns=sensor_columns,
        window_size=window_size,
        stride=stride,
    )

    # =========================================================
    # 10. Generate OFFICIAL TEST windows
    # =========================================================

    X_test, y_test, test_engine_ids = (
        create_sliding_windows(
            test_scaled,
            sensor_columns=sensor_columns,
            window_size=window_size,
            stride=stride,
        )
    )

    return {
        # DataFrames
        "train_df": train_scaled,
        "validation_df": validation_scaled,
        "test_df": test_scaled,

        # Windows
        "X_train": X_train,
        "y_train": y_train,
        "train_engine_ids": train_engine_ids,

        "X_validation": X_validation,
        "y_validation": y_validation,
        "validation_engine_ids": validation_engine_ids,

        "X_test": X_test,
        "y_test": y_test,
        "test_engine_ids": test_engine_ids,

        # Scaler
        "scaler": scaler,

        # Engine IDs
        "train_engines": train_engines,
        "validation_engines": validation_engines,
    }