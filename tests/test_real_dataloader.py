import numpy as np

from src.data.loader import load_dataset
from src.features.dataset_builder import (
    prepare_training_validation_test,
)
from src.models.dataset import (
    create_dataloader,
)


SUBSET = "FD001"

SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
]


def main():

    print("=" * 100)

    print(
        "SensorGuard - Real FD001 PyTorch DataLoader Test"
    )

    print("=" * 100)

    # =========================================================
    # 1. Load real C-MAPSS FD001
    # =========================================================

    print("\n1. Loading FD001")

    dataset = load_dataset(
        SUBSET
    )

    print(
        f"Train rows: {len(dataset['train'])}"
    )

    print(
        f"Test rows: {len(dataset['test'])}"
    )

    print(
        f"RUL values: {len(dataset['rul'])}"
    )

    # =========================================================
    # 2. Run the existing leakage-safe pipeline
    # =========================================================

    print("\n2. Preparing real dataset")

    result = prepare_training_validation_test(
        train_df=dataset["train"],
        test_df=dataset["test"],
        rul_df=dataset["rul"],
        sensor_columns=SENSOR_COLUMNS,
        horizon=30,
        train_ratio=0.85,
        validation_ratio=0.15,
        window_size=30,
        stride=1,
        random_seed=42,
    )

    X_train = result["X_train"]
    y_train = result["y_train"]

    X_validation = result[
        "X_validation"
    ]

    y_validation = result[
        "y_validation"
    ]

    X_test = result["X_test"]
    y_test = result["y_test"]

    # =========================================================
    # 3. Verify NumPy arrays
    # =========================================================

    print("\n3. NumPy Data")

    print("-" * 100)

    print(
        f"X_train shape:       {X_train.shape}"
    )

    print(
        f"y_train shape:       {y_train.shape}"
    )

    print(
        f"X_validation shape:  {X_validation.shape}"
    )

    print(
        f"y_validation shape:  {y_validation.shape}"
    )

    print(
        f"X_test shape:        {X_test.shape}"
    )

    print(
        f"y_test shape:        {y_test.shape}"
    )

    assert X_train.ndim == 3
    assert X_validation.ndim == 3
    assert X_test.ndim == 3

    assert X_train.shape[1:] == (
        30,
        21,
    )

    assert X_validation.shape[1:] == (
        30,
        21,
    )

    assert X_test.shape[1:] == (
        30,
        21,
    )

    assert len(X_train) == len(y_train)
    assert len(X_validation) == len(y_validation)
    assert len(X_test) == len(y_test)

    assert np.isfinite(X_train).all()
    assert np.isfinite(X_validation).all()
    assert np.isfinite(X_test).all()

    print(
        "NumPy validation: PASS"
    )

    # =========================================================
    # 4. Create training DataLoader
    # =========================================================

    print("\n4. Training DataLoader")

    train_loader = create_dataloader(
        X=X_train,
        y=y_train,
        batch_size=64,
        shuffle=True,
        drop_last=False,
    )

    # =========================================================
    # 5. Create validation DataLoader
    # =========================================================

    print("\n5. Validation DataLoader")

    validation_loader = create_dataloader(
        X=X_validation,
        y=y_validation,
        batch_size=64,
        shuffle=False,
        drop_last=False,
    )

    # =========================================================
    # 6. Create test DataLoader
    # =========================================================

    print("\n6. Test DataLoader")

    test_loader = create_dataloader(
        X=X_test,
        y=y_test,
        batch_size=64,
        shuffle=False,
        drop_last=False,
    )

    # =========================================================
    # 7. Inspect training batch
    # =========================================================

    print("\n7. Inspecting Training Batch")

    X_batch, y_batch = next(
        iter(train_loader)
    )

    print(
        f"X batch shape: {tuple(X_batch.shape)}"
    )

    print(
        f"y batch shape: {tuple(y_batch.shape)}"
    )

    print(
        f"X batch dtype:  {X_batch.dtype}"
    )

    print(
        f"y batch dtype:  {y_batch.dtype}"
    )

    assert X_batch.shape[1:] == (
        30,
        21,
    )

    assert y_batch.ndim == 1

    assert X_batch.shape[0] <= 64

    assert X_batch.shape[0] == y_batch.shape[0]

    print(
        "Training batch: PASS"
    )

    # =========================================================
    # 8. Inspect validation batch
    # =========================================================

    print("\n8. Inspecting Validation Batch")

    X_val_batch, y_val_batch = next(
        iter(validation_loader)
    )

    print(
        f"X batch shape: {tuple(X_val_batch.shape)}"
    )

    print(
        f"y batch shape: {tuple(y_val_batch.shape)}"
    )

    assert X_val_batch.shape[1:] == (
        30,
        21,
    )

    assert X_val_batch.shape[0] == (
        y_val_batch.shape[0]
    )

    print(
        "Validation batch: PASS"
    )

    # =========================================================
    # 9. Inspect test batch
    # =========================================================

    print("\n9. Inspecting Test Batch")

    X_test_batch, y_test_batch = next(
        iter(test_loader)
    )

    print(
        f"X batch shape: {tuple(X_test_batch.shape)}"
    )

    print(
        f"y batch shape: {tuple(y_test_batch.shape)}"
    )

    assert X_test_batch.shape[1:] == (
        30,
        21,
    )

    assert X_test_batch.shape[0] == (
        y_test_batch.shape[0]
    )

    print(
        "Test batch: PASS"
    )

    # =========================================================
    # 10. Verify target values
    # =========================================================

    print("\n10. Target Validation")

    train_targets = set(
        y_train.tolist()
    )

    validation_targets = set(
        y_validation.tolist()
    )

    test_targets = set(
        y_test.tolist()
    )

    print(
        f"Train targets:      {sorted(train_targets)}"
    )

    print(
        f"Validation targets: {sorted(validation_targets)}"
    )

    print(
        f"Test targets:       {sorted(test_targets)}"
    )

    assert train_targets.issubset(
        {0, 1}
    )

    assert validation_targets.issubset(
        {0, 1}
    )

    assert test_targets.issubset(
        {0, 1}
    )

    assert train_targets == {
        0,
        1,
    }

    assert validation_targets == {
        0,
        1,
    }

    print(
        "Target validation: PASS"
    )

    # =========================================================
    # 11. Verify complete batch coverage
    # =========================================================

    print("\n11. Batch Coverage")

    train_count = 0

    for X_batch, y_batch in train_loader:

        assert X_batch.shape[0] == (
            y_batch.shape[0]
        )

        train_count += X_batch.shape[0]

    validation_count = 0

    for X_batch, y_batch in validation_loader:

        assert X_batch.shape[0] == (
            y_batch.shape[0]
        )

        validation_count += X_batch.shape[0]

    test_count = 0

    for X_batch, y_batch in test_loader:

        assert X_batch.shape[0] == (
            y_batch.shape[0]
        )

        test_count += X_batch.shape[0]

    print(
        f"Training samples processed:   {train_count}"
    )

    print(
        f"Validation samples processed: {validation_count}"
    )

    print(
        f"Test samples processed:       {test_count}"
    )

    assert train_count == len(X_train)
    assert validation_count == len(X_validation)
    assert test_count == len(X_test)

    print(
        "Batch coverage: PASS"
    )

    # =========================================================
    # 12. Final result
    # =========================================================

    print("\n" + "=" * 100)

    print(
        "REAL FD001 DATALOADER TEST PASSED"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()