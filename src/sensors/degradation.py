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

    The original dataframe is never modified in-place.
    """

    def __init__(
        self,
        random_seed: int = 42,
    ):
        self.random_seed = random_seed

    def _validate_sensor(
        self,
        sensor: str,
        df: pd.DataFrame,
    ) -> None:

        if sensor not in df.columns:
            raise ValueError(
                f"Sensor '{sensor}' not found in dataframe."
            )

        if sensor not in DEFAULT_SENSOR_COLUMNS:
            raise ValueError(
                f"'{sensor}' is not a valid C-MAPSS sensor."
            )

    def random_dropout(
        self,
        df: pd.DataFrame,
        sensors: Optional[List[str]] = None,
        dropout_rate: float = 0.10,
    ) -> pd.DataFrame:
        """
        Randomly replace sensor observations with NaN.

        Parameters
        ----------
        df:
            Input dataframe.

        sensors:
            Sensors to degrade. If None, all 21 sensors
            are considered.

        dropout_rate:
            Fraction of observations to replace with NaN.

        Returns
        -------
        pd.DataFrame
            Degraded copy of the dataframe.
        """

        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(
                "dropout_rate must be between 0 and 1."
            )

        result = df.copy()

        if sensors is None:
            sensors = DEFAULT_SENSOR_COLUMNS.copy()

        rng = np.random.default_rng(
            self.random_seed
        )

        for sensor in sensors:

            self._validate_sensor(
                sensor,
                result,
            )

            n_rows = len(result)

            mask = rng.random(n_rows) < dropout_rate

            result.loc[mask, sensor] = np.nan

        return result