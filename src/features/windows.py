from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
]


def split_engines(
    df: pd.DataFrame,
    train_ratio: float = 0.85,
    validation_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split C-MAPSS training data into training and validation
    partitions at the engine level.

    IMPORTANT:
    This function creates ONLY two partitions:

        1. Training
        2. Validation

    The official C-MAPSS test set must be handled separately.

    No engine can appear in both training and validation.
    """

    # ---------------------------------------------------------
    # Validate ratios
    # ---------------------------------------------------------

    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    if not 0 < validation_ratio < 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    if abs(
        (train_ratio + validation_ratio) - 1.0
    ) > 1e-9:
        raise ValueError(
            "train_ratio + validation_ratio must equal 1."
        )

    # ---------------------------------------------------------
    # Validate required column
    # ---------------------------------------------------------

    if "engine_id" not in df.columns:
        raise ValueError(
            "engine_id column is required."
        )

    # ---------------------------------------------------------
    # Get unique engine IDs
    # ---------------------------------------------------------

    engines = np.array(
        sorted(
            df["engine_id"].unique()
        )
    )

    if len(engines) < 2:
        raise ValueError(
            "At least two engines are required "
            "for train/validation splitting."
        )

    # ---------------------------------------------------------
    # Shuffle engines reproducibly
    # ---------------------------------------------------------

    rng = np.random.default_rng(
        random_seed
    )

    rng.shuffle(engines)

    total_engines = len(engines)

    # ---------------------------------------------------------
    # Calculate training engine count
    # ---------------------------------------------------------

    train_count = int(
        total_engines * train_ratio
    )

    # Ensure both partitions contain engines.
    train_count = max(
        1,
        min(
            train_count,
            total_engines - 1,
        ),
    )

    # ---------------------------------------------------------
    # Engine-level split
    # ---------------------------------------------------------

    train_engines = engines[
        :train_count
    ]

    validation_engines = engines[
        train_count:
    ]

    # ---------------------------------------------------------
    # Create dataframe partitions
    # ---------------------------------------------------------

    train_df = (
        df[
            df["engine_id"].isin(
                train_engines
            )
        ]
        .copy()
        .sort_values(
            ["engine_id", "cycle"]
        )
        .reset_index(drop=True)
    )

    validation_df = (
        df[
            df["engine_id"].isin(
                validation_engines
            )
        ]
        .copy()
        .sort_values(
            ["engine_id", "cycle"]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Final leakage check
    # ---------------------------------------------------------

    train_engine_set = set(
        train_df["engine_id"].unique()
    )

    validation_engine_set = set(
        validation_df["engine_id"].unique()
    )

    overlap = (
        train_engine_set
        & validation_engine_set
    )

    if overlap:
        raise RuntimeError(
            "Engine leakage detected between "
            "training and validation sets: "
            f"{sorted(overlap)}"
        )

    return (
        train_df,
        validation_df,
    )


def verify_engine_split(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> Dict[str, object]:
    """
    Verify that no engine occurs in both
    training and validation.

    The official C-MAPSS test set is intentionally
    not passed to this function because it is an
    independent dataset.
    """

    if "engine_id" not in train_df.columns:
        raise ValueError(
            "train_df must contain 'engine_id'."
        )

    if "engine_id" not in validation_df.columns:
        raise ValueError(
            "validation_df must contain 'engine_id'."
        )

    train_engines = set(
        train_df["engine_id"].unique()
    )

    validation_engines = set(
        validation_df["engine_id"].unique()
    )

    train_validation_overlap = (
        train_engines
        & validation_engines
    )

    valid = (
        len(train_validation_overlap) == 0
    )

    return {
        "valid": valid,
        "train_engines": len(
            train_engines
        ),
        "validation_engines": len(
            validation_engines
        ),
        "train_validation_overlap": sorted(
            train_validation_overlap
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

    Parameters
    ----------
    df:
        Dataframe containing engine_id, cycle, sensors,
        and early_fault.

    sensor_columns:
        Sensor columns to use as model input.

    window_size:
        Number of consecutive cycles in each window.

    stride:
        Step size between consecutive windows.

    Returns
    -------
    X:
        Shape:
        (samples, window_size, sensor_count)

    y:
        Early-fault target corresponding to the
        final timestep of each window.

    engine_ids:
        Engine ID corresponding to each window.
    """

    # ---------------------------------------------------------
    # Validate window parameters
    # ---------------------------------------------------------

    if window_size <= 0:
        raise ValueError(
            "window_size must be greater than zero."
        )

    if stride <= 0:
        raise ValueError(
            "stride must be greater than zero."
        )

    # ---------------------------------------------------------
    # Required dataframe columns
    # ---------------------------------------------------------

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
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    # ---------------------------------------------------------
    # Validate sensor columns
    # ---------------------------------------------------------

    missing_sensors = [
        sensor
        for sensor in sensor_columns
        if sensor not in df.columns
    ]

    if missing_sensors:
        raise ValueError(
            "Missing sensor columns: "
            f"{missing_sensors}"
        )

    # ---------------------------------------------------------
    # Storage
    # ---------------------------------------------------------

    X_windows = []
    y_targets = []
    engine_ids = []

    # ---------------------------------------------------------
    # Process every engine independently
    #
    # This prevents windows from crossing engine boundaries.
    # ---------------------------------------------------------

    for engine_id, engine_df in df.groupby(
        "engine_id",
        sort=True,
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

        # -----------------------------------------------------
        # Skip engines shorter than the window
        # -----------------------------------------------------

        if len(engine_df) < window_size:
            continue

        # -----------------------------------------------------
        # Generate chronological windows
        # -----------------------------------------------------

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

            # Target corresponds to the
            # final timestep of the window.
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

    # ---------------------------------------------------------
    # Handle empty result
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Convert to NumPy arrays
    # ---------------------------------------------------------

    return (
        np.stack(
            X_windows
        ),
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
        np.unique(
            engine_ids
        )
        if len(engine_ids) > 0
        else np.array([])
    )

    positive = int(
        np.sum(
            y == 1
        )
    )

    negative = int(
        np.sum(
            y == 0
        )
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