import numpy as np
import torch

from src.models.dataset import (
    SensorGuardDataset,
    create_dataloader,
)


def create_dummy_data():
    """
    Create deterministic dummy SensorGuard data
    for testing.

    Shape:

        X = (128, 30, 21)
        y = (128,)
    """

    rng = np.random.default_rng(
        seed=42
    )

    X = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(128, 30, 21),
    ).astype(
        np.float32
    )

    y = np.array(
        [0, 1] * 64,
        dtype=np.int64,
    )

    return X, y


def test_dataset_creation():

    X, y = create_dummy_data()

    dataset = SensorGuardDataset(
        X=X,
        y=y,
    )

    assert len(dataset) == 128

    print(
        "Dataset creation: PASS"
    )

    print(
        f"Dataset length: {len(dataset)}"
    )


def test_dataset_sample_shape():

    X, y = create_dummy_data()

    dataset = SensorGuardDataset(
        X=X,
        y=y,
    )

    X_sample, y_sample = dataset[0]

    assert isinstance(
        X_sample,
        torch.Tensor,
    )

    assert isinstance(
        y_sample,
        torch.Tensor,
    )

    assert X_sample.shape == (
        30,
        21,
    )

    assert y_sample.ndim == 0

    print(
        "Dataset sample shape: PASS"
    )

    print(
        f"X sample shape: {tuple(X_sample.shape)}"
    )

    print(
        f"y sample shape: {tuple(y_sample.shape)}"
    )


def test_tensor_dtype():

    X, y = create_dummy_data()

    dataset = SensorGuardDataset(
        X=X,
        y=y,
    )

    X_sample, y_sample = dataset[0]

    assert X_sample.dtype == torch.float32

    assert y_sample.dtype == torch.float32

    print(
        "Tensor dtype: PASS"
    )

    print(
        f"X dtype: {X_sample.dtype}"
    )

    print(
        f"y dtype: {y_sample.dtype}"
    )


def test_dataloader():

    X, y = create_dummy_data()

    loader = create_dataloader(
        X=X,
        y=y,
        batch_size=16,
        shuffle=False,
    )

    X_batch, y_batch = next(
        iter(loader)
    )

    assert X_batch.shape == (
        16,
        30,
        21,
    )

    assert y_batch.shape == (
        16,
    )

    assert X_batch.dtype == torch.float32

    assert y_batch.dtype == torch.float32

    print(
        "DataLoader creation: PASS"
    )

    print(
        f"Batch X shape: {tuple(X_batch.shape)}"
    )

    print(
        f"Batch y shape: {tuple(y_batch.shape)}"
    )


def test_dataloader_all_samples():

    X, y = create_dummy_data()

    loader = create_dataloader(
        X=X,
        y=y,
        batch_size=16,
        shuffle=False,
    )

    sample_count = 0

    for X_batch, y_batch in loader:

        assert X_batch.ndim == 3

        assert y_batch.ndim == 1

        assert X_batch.shape[1:] == (
            30,
            21,
        )

        assert y_batch.shape[0] == (
            X_batch.shape[0]
        )

        sample_count += X_batch.shape[0]

    assert sample_count == 128

    print(
        "DataLoader sample coverage: PASS"
    )

    print(
        f"Samples processed: {sample_count}"
    )


def test_dataloader_shuffle():

    X, y = create_dummy_data()

    loader = create_dataloader(
        X=X,
        y=y,
        batch_size=32,
        shuffle=True,
    )

    X_batch, y_batch = next(
        iter(loader)
    )

    assert X_batch.shape == (
        32,
        30,
        21,
    )

    assert y_batch.shape == (
        32,
    )

    print(
        "DataLoader shuffle mode: PASS"
    )


def test_invalid_sample_count():

    X, y = create_dummy_data()

    invalid_y = y[:100]

    try:

        SensorGuardDataset(
            X=X,
            y=invalid_y,
        )

    except ValueError:

        print(
            "Sample-count validation: PASS"
        )

        return

    raise AssertionError(
        "Dataset should reject X and y "
        "with different sample counts."
    )


def test_invalid_target():

    X, y = create_dummy_data()

    invalid_y = y.copy()

    invalid_y[0] = 2

    try:

        SensorGuardDataset(
            X=X,
            y=invalid_y,
        )

    except ValueError:

        print(
            "Target validation: PASS"
        )

        return

    raise AssertionError(
        "Dataset should reject non-binary targets."
    )


def test_non_finite_input():

    X, y = create_dummy_data()

    X = X.copy()

    X[0, 0, 0] = np.nan

    try:

        SensorGuardDataset(
            X=X,
            y=y,
        )

    except ValueError:

        print(
            "Finite-value validation: PASS"
        )

        return

    raise AssertionError(
        "Dataset should reject NaN values."
    )


if __name__ == "__main__":

    print("=" * 80)

    print(
        "SensorGuard - PyTorch Dataset/DataLoader Tests"
    )

    print("=" * 80)

    test_dataset_creation()

    test_dataset_sample_shape()

    test_tensor_dtype()

    test_dataloader()

    test_dataloader_all_samples()

    test_dataloader_shuffle()

    test_invalid_sample_count()

    test_invalid_target()

    test_non_finite_input()

    print("=" * 80)

    print(
        "ALL DATASET/DATALOADER TESTS PASSED"
    )

    print("=" * 80)