
from typing import List, Optional

import numpy as np
import pandas as pd


DEFAULT_SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
]


class SensorDegradationEngine:
    """
    Controlled synthetic sensor degradation engine.

    Supported mechanisms:
        1. Random dropout
        2. Missing intervals
        3. Additive Gaussian noise
        4. Persistent bias
        5. Progressive drift

    The input DataFrame is not modified in-place.

    Temporal degradation mechanisms operate independently
    within each engine trajectory.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    # ============================================================
    # Common Validation
    # ============================================================

    def _validate_sensor(
        self,
        sensor: str,
        df: pd.DataFrame,
    ) -> None:
        """Validate a C-MAPSS sensor column."""

        if sensor not in df.columns:
            raise ValueError(
                f"Sensor '{sensor}' not found in dataframe."
            )

        if sensor not in DEFAULT_SENSOR_COLUMNS:
            raise ValueError(
                f"'{sensor}' is not a valid C-MAPSS sensor."
            )

    def _validate_engine_column(
        self,
        df: pd.DataFrame,
    ) -> None:
        """Require valid engine IDs for temporal degradation."""

        if "engine_id" not in df.columns:
            raise ValueError(
                "Dataframe must contain an 'engine_id' "
                "column for engine-aware degradation."
            )

        if df["engine_id"].isna().any():
            raise ValueError(
                "engine_id cannot contain missing values."
            )

    def _resolve_sensors(
        self,
        df: pd.DataFrame,
        sensors: Optional[List[str]],
    ) -> List[str]:
        """Resolve and validate selected sensor columns."""

        selected = (
            DEFAULT_SENSOR_COLUMNS.copy()
            if sensors is None
            else list(sensors)
        )

        if not selected:
            raise ValueError(
                "At least one sensor must be selected."
            )

        if len(selected) != len(set(selected)):
            raise ValueError(
                "Duplicate sensor names are not allowed."
            )

        for sensor in selected:
            self._validate_sensor(sensor, df)

        return selected

    @staticmethod
    def _validate_finite_nonnegative(
        name: str,
        value: float,
    ) -> None:
        """Validate a finite, non-negative numeric parameter."""

        try:
            valid = bool(
                np.isfinite(value) and value >= 0
            )
        except (TypeError, ValueError):
            valid = False

        if not valid:
            raise ValueError(
                f"{name} must be a finite, "
                "non-negative number."
            )

    # ============================================================
    # Random Dropout
    # ============================================================

    def random_dropout(
        self,
        df: pd.DataFrame,
        sensors: Optional[List[str]] = None,
        dropout_rate: float = 0.10,
    ) -> pd.DataFrame:
        """
        Randomly replace selected sensor observations with NaN.

        dropout_rate must be between 0 and 1, inclusive.
        """

        try:
            valid_rate = bool(
                np.isfinite(dropout_rate)
                and 0.0 <= dropout_rate <= 1.0
            )
        except (TypeError, ValueError):
            valid_rate = False

        if not valid_rate:
            raise ValueError(
                "dropout_rate must be between 0 and 1."
            )

        selected = self._resolve_sensors(df, sensors)
        result = df.copy()
        rng = np.random.default_rng(self.random_seed)

        for sensor in selected:
            mask = (
                rng.random(len(result))
                < dropout_rate
            )
            result.loc[mask, sensor] = np.nan

        return result

    # ============================================================
    # Non-Overlapping Interval Sampling
    # ============================================================

    def _sample_non_overlapping_intervals(
        self,
        n_rows: int,
        interval_length: int,
        num_intervals: int,
        rng: np.random.Generator,
    ) -> List[tuple]:
        """
        Generate non-overlapping intervals within one engine.

        Returns (start, end) positions; end is exclusive.
        """

        if interval_length <= 0:
            raise ValueError(
                "interval_length must be greater than 0."
            )

        if num_intervals <= 0:
            raise ValueError(
                "num_intervals must be greater than 0."
            )

        if interval_length > n_rows:
            raise ValueError(
                "interval_length cannot exceed "
                "engine trajectory length."
            )

        max_possible = n_rows // interval_length

        if num_intervals > max_possible:
            raise ValueError(
                f"Cannot place {num_intervals} non-overlapping "
                f"intervals of length {interval_length} "
                f"inside a trajectory of length {n_rows}."
            )

        possible_starts = np.arange(
            0,
            n_rows - interval_length + 1,
        )
        shuffled_starts = rng.permutation(possible_starts)

        selected = []

        for start_value in shuffled_starts:
            start = int(start_value)
            end = start + interval_length

            overlaps = any(
                not (
                    end <= existing_start
                    or start >= existing_end
                )
                for existing_start, existing_end in selected
            )

            if not overlaps:
                selected.append((start, end))

            if len(selected) == num_intervals:
                break

        if len(selected) != num_intervals:
            raise ValueError(
                "Unable to generate the requested "
                "number of non-overlapping intervals."
            )

        selected.sort()
        return selected

    # ============================================================
    # Missing Interval
    # ============================================================

    def missing_interval(
        self,
        df: pd.DataFrame,
        sensors: Optional[List[str]] = None,
        interval_length: int = 10,
        num_intervals: int = 1,
    ) -> pd.DataFrame:
        """
        Replace contiguous sensor observations with NaN.

        Intervals are generated independently for each engine.
        The same intervals are applied to selected sensors.

        num_intervals is the number of intervals per engine.
        """

        if (
            not isinstance(interval_length, (int, np.integer))
            or isinstance(interval_length, (bool, np.bool_))
            or interval_length <= 0
        ):
            raise ValueError(
                "interval_length must be a positive integer."
            )

        if (
            not isinstance(num_intervals, (int, np.integer))
            or isinstance(num_intervals, (bool, np.bool_))
            or num_intervals <= 0
        ):
            raise ValueError(
                "num_intervals must be a positive integer."
            )

        self._validate_engine_column(df)
        selected = self._resolve_sensors(df, sensors)

        result = df.copy()
        rng = np.random.default_rng(self.random_seed)

        for _, engine_group in result.groupby(
            "engine_id",
            sort=False,
        ):
            engine_indices = engine_group.index
            n_engine_rows = len(engine_group)

            intervals = self._sample_non_overlapping_intervals(
                n_rows=n_engine_rows,
                interval_length=interval_length,
                num_intervals=num_intervals,
                rng=rng,
            )

            for start, end in intervals:
                target_indices = engine_indices[start:end]

                for sensor in selected:
                    result.loc[
                        target_indices, sensor
                    ] = np.nan

        return result

    # ============================================================
    # Additive Gaussian Noise
    # ============================================================

    def add_noise(
        self,
        df: pd.DataFrame,
        sensors: Optional[List[str]] = None,
        noise_std: float = 0.10,
    ) -> pd.DataFrame:
        """
        Add zero-mean Gaussian noise to selected sensors.

        noise_std is a multiplier of the sensor's population
        standard deviation.

        Example:
            noise_std=0.20 means noise sigma is 20% of
            the sensor's standard deviation.
        """

        self._validate_finite_nonnegative(
            "noise_std",
            noise_std,
        )

        selected = self._resolve_sensors(df, sensors)
        result = df.copy()
        rng = np.random.default_rng(self.random_seed)

        for sensor in selected:
            values = pd.to_numeric(
                result[sensor],
                errors="raise",
            )

            sensor_std = values.std(ddof=0)

            if not np.isfinite(sensor_std) or sensor_std == 0:
                continue

            sigma = noise_std * sensor_std

            noise = rng.normal(
                loc=0.0,
                scale=sigma,
                size=len(result),
            )

            result[sensor] = (
                values.to_numpy() + noise
            )

        return result

    # ============================================================
    # Persistent Bias
    # ============================================================

    def add_bias(
        self,
        df: pd.DataFrame,
        sensors: Optional[List[str]] = None,
        bias_scale: float = 0.20,
        direction: str = "positive",
    ) -> pd.DataFrame:
        """
        Add a persistent additive offset to selected sensors.

        bias = direction_sign * bias_scale * sensor_std

        direction:
            positive -> positive offset
            negative -> negative offset
            random   -> randomly select a sign per sensor

        Standard deviation is calculated over the supplied
        DataFrame for each selected sensor.
        """

        self._validate_finite_nonnegative(
            "bias_scale",
            bias_scale,
        )

        valid_directions = {
            "positive",
            "negative",
            "random",
        }

        if direction not in valid_directions:
            raise ValueError(
                "direction must be 'positive', "
                "'negative', or 'random'."
            )

        selected = self._resolve_sensors(df, sensors)
        result = df.copy()
        rng = np.random.default_rng(self.random_seed)

        for sensor in selected:
            values = pd.to_numeric(
                result[sensor],
                errors="raise",
            )

            sensor_std = values.std(ddof=0)

            if not np.isfinite(sensor_std) or sensor_std == 0:
                continue

            if direction == "positive":
                sign = 1.0
            elif direction == "negative":
                sign = -1.0
            else:
                sign = float(rng.choice([-1.0, 1.0]))

            bias = sign * bias_scale * sensor_std

            result[sensor] = (
                values.to_numpy() + bias
            )

        return result

    
    
    # ============================================================
    # Progressive Drift
    # ============================================================

    def add_drift(
        self,
        df: pd.DataFrame,
        sensors: Optional[List[str]] = None,
        drift_scale: float = 0.50,
        direction: str = "positive",
    ) -> pd.DataFrame:
        """
        Add linear synthetic drift independently to each engine.

        Drift starts at zero and increases linearly until the
        final observed row of each engine trajectory.

        Final drift magnitude:
            sign * drift_scale * engine_sensor_std

        direction:
            positive -> increasing additive offset
            negative -> decreasing additive offset
            random   -> randomly select a sign per sensor

        Requires engine_id. Rows within each engine must be
        ordered chronologically before applying drift.
        """

        self._validate_finite_nonnegative(
            "drift_scale",
            drift_scale,
        )

        valid_directions = {
            "positive",
            "negative",
            "random",
        }

        if direction not in valid_directions:
            raise ValueError(
                "direction must be 'positive', "
                "'negative', or 'random'."
            )

        self._validate_engine_column(df)
        selected = self._resolve_sensors(df, sensors)

        result = df.copy()

        # IMPORTANT:
        # Drift produces floating-point values. Convert selected
        # sensor columns before using label-based assignment.
        for sensor in selected:
            result[sensor] = pd.to_numeric(
                result[sensor],
                errors="raise",
            ).astype(float)

        rng = np.random.default_rng(self.random_seed)

        for _, engine_group in result.groupby(
            "engine_id",
            sort=False,
        ):
            engine_indices = engine_group.index
            n_engine_rows = len(engine_group)

            if n_engine_rows <= 1:
                continue

            progress = np.linspace(
                0.0,
                1.0,
                n_engine_rows,
            )

            for sensor in selected:
                values = engine_group[sensor]
                sensor_std = values.std(ddof=0)

                if (
                    not np.isfinite(sensor_std)
                    or sensor_std == 0
                ):
                    continue

                if direction == "positive":
                    sign = 1.0
                elif direction == "negative":
                    sign = -1.0
                else:
                    sign = float(
                        rng.choice([-1.0, 1.0])
                    )

                final_drift = (
                    sign * drift_scale * sensor_std
                )

                drift_values = (
                    progress * final_drift
                )

                result.loc[
                    engine_indices,
                    sensor,
                ] = (
                    values.to_numpy() + drift_values
                )

        return result
