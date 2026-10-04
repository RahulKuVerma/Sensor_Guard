from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
]


def split_engines(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split C-MAPSS data by engine.

    No engine can appear in more than one split.
    """

    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    if not 0 < validation_ratio < 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "train_ratio + validation_ratio must be < 1."
        )

    if "engine_id" not in df.columns:
        raise ValueError(
            "engine_id column is required."
        )

    engines = np.array(
        sorted(df["engine_id"].unique())
    )

    rng = np.random.default_rng(
        random_seed
    )

    rng.shuffle(engines)

    total_engines = len(engines)

    train_end = int(
        total_engines * train_ratio
    )

    validation_end = (
        train_end
        + int(
            total_engines
            * validation_ratio
        )
    )

    train_engines = engines[
        :train_end
    ]

    validation_engines = engines[
        train_end:validation_end
    ]

    test_engines = engines[
        validation_end:
    ]

    train_df = df[
        df["engine_id"].isin(
            train_engines
        )
    ].copy()

    validation_df = df[
        df["engine_id"].isin(
            validation_engines
        )
    ].copy()

    test_df = df[
        df["engine_id"].isin(
            test_engines
        )
    ].copy()

    return (
        train_df.sort_values(
            ["engine_id", "cycle"]
        ).reset_index(drop=True),
        validation_df.sort_values(
            ["engine_id", "cycle"]
        ).reset_index(drop=True),
        test_df.sort_values(
            ["engine_id", "cycle"]
        ).reset_index(drop=True),
    )


def verify_engine_split(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, object]:
    """
    Verify that no engine occurs in multiple splits.
    """

    train_engines = set(
        train_df["engine_id"].unique()
    )

    validation_engines = set(
        validation_df["engine_id"].unique()
    )

    test_engines = set(
        test_df["engine_id"].unique()
    )

    train_validation_overlap = (
        train_engines
        & validation_engines
    )

    train_test_overlap = (
        train_engines
        & test_engines
    )

    validation_test_overlap = (
        validation_engines
        & test_engines
    )

    valid = (
        len(train_validation_overlap) == 0
        and len(train_test_overlap) == 0
        and len(validation_test_overlap) == 0
    )

    return {
        "valid": valid,
        "train_engines": len(
            train_engines
        ),
        "validation_engines": len(
            validation_engines
        ),
        "test_engines": len(
            test_engines
        ),
        "train_validation_overlap": sorted(
            train_validation_overlap
        ),
        "train_test_overlap": sorted(
            train_test_overlap
        ),
        "validation_test_overlap": sorted(
            validation_test_overlap
        ),
    }


def create_sliding_windows(
    df: pd.DataFrame,
    sensor_columns: List[str],
    window_size: int = 30,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create chronological sliding windows independently
    for every engine.

    Returns:
        X:
            Shape = (samples, window_size, sensors)

        y:
            Early-fault target for the final timestep.

        engine_ids:
            Engine ID corresponding to each window.
    """

    if window_size <= 0:
        raise ValueError(
            "window_size must be greater than zero."
        )

    if stride <= 0:
        raise ValueError(
            "stride must be greater than zero."
        )

    required_columns = {
        "engine_id",
        "cycle",
        "early_fault",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    missing_sensors = [
        sensor
        for sensor in sensor_columns
        if sensor not in df.columns
    ]

    if missing_sensors:
        raise ValueError(
            f"Missing sensor columns: "
            f"{missing_sensors}"
        )

    X_windows = []
    y_targets = []
    engine_ids = []

    for engine_id, engine_df in (
        df.groupby(
            "engine_id",
            sort=True,
        )
    ):

        engine_df = (
            engine_df
            .sort_values("cycle")
            .reset_index(drop=True)
        )

        sensor_values = (
            engine_df[
                sensor_columns
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        targets = (
            engine_df[
                "early_fault"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        if len(engine_df) < window_size:
            continue

        for start in range(
            0,
            len(engine_df)
            - window_size
            + 1,
            stride,
        ):

            end = (
                start
                + window_size
            )

            window = sensor_values[
                start:end
            ]

            target = targets[
                end - 1
            ]

            X_windows.append(
                window
            )

            y_targets.append(
                target
            )

            engine_ids.append(
                engine_id
            )

    if not X_windows:
        return (
            np.empty(
                (
                    0,
                    window_size,
                    len(sensor_columns),
                ),
                dtype=np.float32,
            ),
            np.empty(
                (0,),
                dtype=np.int64,
            ),
            np.empty(
                (0,),
                dtype=np.int64,
            ),
        )

    return (
        np.stack(X_windows),
        np.asarray(
            y_targets,
            dtype=np.int64,
        ),
        np.asarray(
            engine_ids,
            dtype=np.int64,
        ),
    )


def summarize_windows(
    X: np.ndarray,
    y: np.ndarray,
    engine_ids: np.ndarray,
) -> Dict[str, object]:
    """
    Generate window-level statistics.
    """

    unique_engines = (
        np.unique(engine_ids)
        if len(engine_ids) > 0
        else np.array([])
    )

    positive = int(
        np.sum(y == 1)
    )

    negative = int(
        np.sum(y == 0)
    )

    total = len(y)

    return {
        "samples": total,
        "window_size": (
            X.shape[1]
            if X.ndim == 3
            else 0
        ),
        "sensor_count": (
            X.shape[2]
            if X.ndim == 3
            else 0
        ),
        "engines": len(
            unique_engines
        ),
        "normal_windows": negative,
        "early_fault_windows": positive,
        "early_fault_ratio": (
            positive / total
            if total > 0
            else 0.0
        ),
    }