from pathlib import Path

import numpy as np
import torch

from src.data.loader import load_dataset
from src.features.dataset_builder import (
    prepare_training_validation_test,
)
from src.models.baseline import (
    create_baseline_model,
)
from src.models.dataset import (
    create_dataloader,
)
from src.models.trainer import (
    BaselineTrainer,
)


# =============================================================
# Configuration
# =============================================================

SUBSET = "FD001"

HORIZON = 30
WINDOW_SIZE = 30
STRIDE = 1

TRAIN_RATIO = 0.85
VALIDATION_RATIO = 0.15

BATCH_SIZE = 64

EPOCHS = 50
PATIENCE = 8

LEARNING_RATE = 0.001

RANDOM_SEED = 42

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.30

SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
]

CHECKPOINT_PATH = (
    Path("results")
    / "models"
    / SUBSET
    / "baseline_lstm.pt"
)


# =============================================================
# Reproducibility
# =============================================================

def set_seed(seed: int) -> None:

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================
# Calculate positive-class weight
# =============================================================

def calculate_positive_class_weight(
    y_train: np.ndarray,
) -> float:

    negative_count = int(
        np.sum(y_train == 0)
    )

    positive_count = int(
        np.sum(y_train == 1)
    )

    if positive_count == 0:
        raise ValueError(
            "Training data contains no positive "
            "early-fault samples."
        )

    if negative_count == 0:
        raise ValueError(
            "Training data contains no negative "
            "normal samples."
        )

    return (
        negative_count
        / positive_count
    )


# =============================================================
# Main
# =============================================================

def main():

    print("=" * 100)

    print(
        "SensorGuard - Clean Baseline Training"
    )

    print("=" * 100)

    set_seed(
        RANDOM_SEED
    )

    # =========================================================
    # 1. Device
    # =========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # =========================================================
    # 2. Load FD001
    # =========================================================

    print("\nLoading FD001...")

    dataset = load_dataset(
        SUBSET
    )

    print(
        f"Training rows: "
        f"{len(dataset['train'])}"
    )

    print(
        f"Test rows: "
        f"{len(dataset['test'])}"
    )

    print(
        f"RUL values: "
        f"{len(dataset['rul'])}"
    )

    # =========================================================
    # 3. Prepare dataset
    # =========================================================

    print(
        "\nPreparing leakage-safe dataset..."
    )

    result = (
        prepare_training_validation_test(
            train_df=dataset["train"],
            test_df=dataset["test"],
            rul_df=dataset["rul"],
            sensor_columns=SENSOR_COLUMNS,
            horizon=HORIZON,
            train_ratio=TRAIN_RATIO,
            validation_ratio=VALIDATION_RATIO,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            random_seed=RANDOM_SEED,
        )
    )

    X_train = result[
        "X_train"
    ]

    y_train = result[
        "y_train"
    ]

    X_validation = result[
        "X_validation"
    ]

    y_validation = result[
        "y_validation"
    ]

    X_test = result[
        "X_test"
    ]

    y_test = result[
        "y_test"
    ]

    # =========================================================
    # 4. Dataset summary
    # =========================================================

    print("\nDataset Summary")
    print("-" * 100)

    print(
        f"X_train:       {X_train.shape}"
    )

    print(
        f"y_train:       {y_train.shape}"
    )

    print(
        f"X_validation:  {X_validation.shape}"
    )

    print(
        f"y_validation:  {y_validation.shape}"
    )

    print(
        f"X_test:        {X_test.shape}"
    )

    print(
        f"y_test:        {y_test.shape}"
    )

    # =========================================================
    # 5. Training class distribution
    # =========================================================

    negative_count = int(
        np.sum(y_train == 0)
    )

    positive_count = int(
        np.sum(y_train == 1)
    )

    positive_ratio = (
        positive_count
        / len(y_train)
    )

    positive_class_weight = (
        calculate_positive_class_weight(
            y_train
        )
    )

    print("\nTraining Class Distribution")
    print("-" * 100)

    print(
        f"Normal samples:      "
        f"{negative_count}"
    )

    print(
        f"Early-fault samples: "
        f"{positive_count}"
    )

    print(
        f"Early-fault ratio:   "
        f"{positive_ratio:.4f}"
    )

    print(
        f"Positive class weight:"
        f" {positive_class_weight:.6f}"
    )

    # =========================================================
    # 6. Create DataLoaders
    # =========================================================

    print("\nCreating DataLoaders...")

    train_loader = create_dataloader(
        X=X_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    validation_loader = create_dataloader(
        X=X_validation,
        y=y_validation,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    test_loader = create_dataloader(
        X=X_test,
        y=y_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    print(
        f"Training batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(validation_loader)}"
    )

    print(
        f"Test batches: "
        f"{len(test_loader)}"
    )

    # =========================================================
    # 7. Create model
    # =========================================================

    print("\nCreating LSTM model...")

    model = create_baseline_model(
        input_size=21,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    print(model)

    # =========================================================
    # 8. Create trainer
    # =========================================================

    trainer = BaselineTrainer(
        model=model,
        learning_rate=LEARNING_RATE,
        positive_class_weight=(
            positive_class_weight
        ),
        device=device,
    )

    # =========================================================
    # 9. Train
    # =========================================================

    print("\nStarting baseline training...")
    print("=" * 100)

    history = trainer.fit(
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=EPOCHS,
        patience=PATIENCE,
        checkpoint_path=CHECKPOINT_PATH,
    )

    # =========================================================
    # 10. Training summary
    # =========================================================

    print("\n" + "=" * 100)

    print(
        "Training Completed"
    )

    print("=" * 100)

    print(
        f"Best epoch: "
        f"{trainer.best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{trainer.best_validation_loss:.6f}"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    print(
        f"Epochs completed: "
        f"{len(history['train_loss'])}"
    )

    # =========================================================
    # 11. Load best checkpoint
    # =========================================================

    print(
        "\nLoading best checkpoint..."
    )

    trainer.load_checkpoint(
        CHECKPOINT_PATH
    )

    print(
        "Best checkpoint loaded successfully."
    )

    # =========================================================
    # 12. Final validation
    # =========================================================

    validation_metrics = (
        trainer.validate(
            validation_loader
        )
    )

    print("\nBest Model Validation")
    print("-" * 100)

    print(
        f"Loss:      "
        f"{validation_metrics['loss']:.6f}"
    )

    print(
        f"Accuracy:  "
        f"{validation_metrics['accuracy']:.6f}"
    )

    print(
        f"Precision: "
        f"{validation_metrics['precision']:.6f}"
    )

    print(
        f"Recall:    "
        f"{validation_metrics['recall']:.6f}"
    )

    print(
        f"F1:        "
        f"{validation_metrics['f1']:.6f}"
    )

    # =========================================================
    # 13. Test data is intentionally NOT evaluated here
    # =========================================================

    print(
        "\nOfficial test set has been loaded but "
        "is intentionally NOT used for model selection."
    )

    print(
        "Final test evaluation will be implemented "
        "in the dedicated evaluation step."
    )

    print("\n" + "=" * 100)

    print(
        "BASELINE TRAINING PIPELINE COMPLETE"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()