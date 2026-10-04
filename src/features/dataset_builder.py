from typing import Dict, List

import pandas as pd

from src.data.preprocessing import create_training_dataset
from src.data.test_targets import create_test_targets
from src.features.windows import (
    create_sliding_windows,
    split_engines,
    verify_engine_split,
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

    # ---------------------------------------------------------
    # 1. Prepare training labels
    # ---------------------------------------------------------

    training = create_training_dataset(
        train_df,
        horizon=horizon,
    )

    # ---------------------------------------------------------
    # 2. Split ONLY the original training trajectories
    #    into training and validation engines.
    # ---------------------------------------------------------

    training_split, validation_split, unused_test_split = split_engines(
        training,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        random_seed=random_seed,
    )

    # The previous split function returns three partitions.
    # We intentionally do NOT use its third partition as final test.
    #
    # For this pipeline:
    #   training_split    -> model training
    #   validation_split  -> model selection
    #   official test set -> final evaluation
    #
    # To avoid silently discarding engines, combine the third partition
    # back into training data.

    training_engines = set(
        training_split["engine_id"].unique()
    )

    unused_test_engines = set(
        unused_test_split["engine_id"].unique()
    )

    training_split = pd.concat(
        [
            training_split,
            unused_test_split,
        ],
        ignore_index=True,
    ).sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 3. Validate training/validation separation
    # ---------------------------------------------------------

    train_validation_check = verify_engine_split(
        training_split,
        validation_split,
        pd.DataFrame(
            {"engine_id": []}
        ),
    )

    if not train_validation_check["valid"]:
        raise RuntimeError(
            "Training/validation engine leakage detected."
        )

    # ---------------------------------------------------------
    # 4. Construct official test targets
    # ---------------------------------------------------------

    official_test = create_test_targets(
        test_df=test_df,
        rul_df=rul_df,
        horizon=horizon,
    )

    # ---------------------------------------------------------
    # 5. Create windows
    # ---------------------------------------------------------

    X_train, y_train, train_engine_ids = create_sliding_windows(
        training_split,
        sensor_columns=sensor_columns,
        window_size=window_size,
        stride=stride,
    )

    X_validation, y_validation, validation_engine_ids = (
        create_sliding_windows(
            validation_split,
            sensor_columns=sensor_columns,
            window_size=window_size,
            stride=stride,
        )
    )

    X_test, y_test, test_engine_ids = create_sliding_windows(
        official_test,
        sensor_columns=sensor_columns,
        window_size=window_size,
        stride=stride,
    )

    return {
        "train_df": training_split,
        "validation_df": validation_split,
        "test_df": official_test,

        "X_train": X_train,
        "y_train": y_train,
        "train_engine_ids": train_engine_ids,

        "X_validation": X_validation,
        "y_validation": y_validation,
        "validation_engine_ids": validation_engine_ids,

        "X_test": X_test,
        "y_test": y_test,
        "test_engine_ids": test_engine_ids,
    }