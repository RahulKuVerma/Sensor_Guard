from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
]


class SensorScaler:
    """
    Leakage-safe StandardScaler for SensorGuard.

    The scaler MUST be fitted only on training data.
    Validation and test data are transformed using the
    already-fitted training scaler.
    """

    def __init__(self, sensor_columns: List[str] | None = None):
        self.sensor_columns = (
            sensor_columns
            if sensor_columns is not None
            else DEFAULT_SENSOR_COLUMNS.copy()
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "SensorScaler":
        """
        Fit scaler using training data only.
        """
        missing = [
            column
            for column in self.sensor_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing sensor columns during scaler fitting: {missing}"
            )

        if len(df) == 0:
            raise ValueError("Cannot fit scaler on an empty dataframe.")

        self.scaler.fit(df[self.sensor_columns])
        self.is_fitted = True

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using the training-fitted scaler.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Scaler has not been fitted. "
                "Call fit() using training data first."
            )

        missing = [
            column
            for column in self.sensor_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing sensor columns during transformation: {missing}"
            )

        result = df.copy()

        result[self.sensor_columns] = self.scaler.transform(
            result[self.sensor_columns]
        )

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on training data and transform it.
        """
        self.fit(df)
        return self.transform(df)

    def save(self, path: str | Path) -> None:
        """
        Save fitted scaler to disk.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Cannot save an unfitted scaler."
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "sensor_columns": self.sensor_columns,
                "scaler": self.scaler,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SensorScaler":
        """
        Load a previously fitted scaler.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Scaler file not found: {path}"
            )

        payload = joblib.load(path)

        instance = cls(
            sensor_columns=payload["sensor_columns"]
        )
        instance.scaler = payload["scaler"]
        instance.is_fitted = True

        return instance