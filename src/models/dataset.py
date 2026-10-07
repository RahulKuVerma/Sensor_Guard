import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SensorGuardDataset(Dataset):
    """
    PyTorch Dataset for SensorGuard early-fault prediction.

    X:
        Shape:
            (samples, sequence_length, sensor_count)

        Example:
            (15102, 30, 21)

    y:
        Shape:
            (samples,)

        Binary target:
            0 = normal
            1 = early fault
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        if not isinstance(X, np.ndarray):
            raise TypeError(
                "X must be a NumPy array."
            )

        if not isinstance(y, np.ndarray):
            raise TypeError(
                "y must be a NumPy array."
            )

        if X.ndim != 3:
            raise ValueError(
                "X must be a 3D array with shape "
                "(samples, sequence_length, sensor_count). "
                f"Received shape: {X.shape}"
            )

        if y.ndim != 1:
            raise ValueError(
                "y must be a 1D array with shape "
                "(samples,). "
                f"Received shape: {y.shape}"
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number "
                f"of samples. X={len(X)}, y={len(y)}"
            )

        if len(X) == 0:
            raise ValueError(
                "Dataset cannot be empty."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "X contains NaN or infinite values."
            )

        if not np.isfinite(y).all():
            raise ValueError(
                "y contains NaN or infinite values."
            )

        unique_targets = set(
            np.unique(y).tolist()
        )

        if not unique_targets.issubset({0, 1}):
            raise ValueError(
                "y must contain only binary targets "
                "{0, 1}. "
                f"Received: {sorted(unique_targets)}"
            )

        # -----------------------------------------------------
        # Convert arrays to PyTorch tensors.
        #
        # Float32 is used because it is the standard dtype
        # for neural-network inputs and works efficiently
        # with PyTorch.
        # -----------------------------------------------------

        self.X = torch.from_numpy(
            X.astype(
                np.float32,
                copy=False,
            )
        )

        self.y = torch.from_numpy(
            y.astype(
                np.float32,
                copy=False,
            )
        )

    def __len__(self) -> int:
        """
        Return the number of samples.
        """
        return len(self.X)

    def __getitem__(
        self,
        index: int,
    ):
        """
        Return one sample.

        Returns:
            X_sample:
                Shape = (sequence_length, sensor_count)

            y_sample:
                Scalar float tensor.
        """

        return (
            self.X[index],
            self.y[index],
        )


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for SensorGuard.

    Parameters
    ----------
    X:
        Input windows.

    y:
        Binary targets.

    batch_size:
        Number of windows per batch.

    shuffle:
        Whether to shuffle samples.

        Training:
            True

        Validation/Test:
            False

    drop_last:
        Whether to discard the final incomplete batch.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    dataset = SensorGuardDataset(
        X=X,
        y=y,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )