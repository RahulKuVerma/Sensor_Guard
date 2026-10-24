
"""
SensorGuard — Train additional C-MAPSS subsets.

Trains FD002, FD003 and FD004 using the existing pipeline.
FD001 is deliberately excluded and its checkpoint is untouched.

Protocol:
- Engine-level 85/15 training-validation split.
- StandardScaler fitted on training engines only.
- Positive-class weight computed from training-window labels only.
- Validation loss controls early stopping and checkpoint selection.
- Official test labels are never used for training or model selection.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.loader import load_dataset
from src.features.dataset_builder import (
    prepare_training_validation_test,
)
from src.models.baseline import create_baseline_model
from src.models.dataset import create_dataloader
from src.models.trainer import BaselineTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBSETS = ["FD002", "FD003", "FD004"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]

SEED = 42
HORIZON = 30
WINDOW_SIZE = 30
STRIDE = 1

TRAIN_RATIO = 0.85
VALIDATION_RATIO = 0.15

BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 8
LEARNING_RATE = 0.001

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.30


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_positive_class_weight(y_train: np.ndarray) -> float:
    """
    Calculate negative/positive ratio using training labels only.
    """
    labels = np.asarray(y_train).reshape(-1)

    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))

    if positives == 0:
        raise ValueError(
            "Training labels contain no positive examples."
        )

    if negatives == 0:
        raise ValueError(
            "Training labels contain no negative examples."
        )

    return negatives / positives


def train_subset(subset: str, force: bool = False) -> dict:
    """Train one subset and save its checkpoint and history."""

    set_seed(SEED)

    checkpoint_dir = PROJECT_ROOT / "results" / "models" / subset
    checkpoint_path = checkpoint_dir / "baseline_lstm.pt"

    output_dir = PROJECT_ROOT / "results" / "metrics" / subset
    history_path = output_dir / "training_history.csv"
    report_path = output_dir / "training_report.json"

    if checkpoint_path.exists() and not force:
        print(
            f"\n[SKIP] {subset}: checkpoint already exists:\n"
            f"       {checkpoint_path}\n"
            "       Use --force only if you intend to retrain it."
        )
        return {
            "subset": subset,
            "status": "skipped_existing_checkpoint",
            "checkpoint": str(checkpoint_path),
        }

    print("\n" + "=" * 76)
    print(f"Training SensorGuard baseline: {subset}")
    print("=" * 76)

    # Load and sort this subset's data.
    dataset = load_dataset(subset)

    train_df = dataset["train"].copy()
    test_df = dataset["test"].copy()
    rul_df = dataset["rul"].copy()

    train_df = train_df.sort_values(
        ["engine_id", "cycle"], kind="stable"
    ).reset_index(drop=True)

    test_df = test_df.sort_values(
        ["engine_id", "cycle"], kind="stable"
    ).reset_index(drop=True)

    # Reuse the project's established split, scaler and window pipeline.
    prepared = prepare_training_validation_test(
        train_df=train_df,
        test_df=test_df,
        rul_df=rul_df,
        sensor_columns=SENSOR_COLUMNS,
        horizon=HORIZON,
        train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        random_seed=SEED,
    )

    X_train = np.asarray(prepared["X_train"])
    y_train = np.asarray(prepared["y_train"]).astype(int)

    X_val = np.asarray(prepared["X_validation"])
    y_val = np.asarray(prepared["y_validation"]).astype(int)

    # Defensive checks before training.
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError(f"{subset}: training or validation windows are empty.")

    if not np.isfinite(X_train).all():
        raise ValueError(f"{subset}: training windows contain non-finite values.")

    if not np.isfinite(X_val).all():
        raise ValueError(f"{subset}: validation windows contain non-finite values.")

    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError(
            f"{subset}: training labels must contain both classes."
        )

    if set(np.unique(y_val)) != {0, 1}:
        raise ValueError(
            f"{subset}: validation labels must contain both classes."
        )

    # Compute the class weight using training labels only.
    positive_weight = calculate_positive_class_weight(y_train)

    print(f"Training windows:   {len(X_train):,}")
    print(f"Validation windows: {len(X_val):,}")
    print(f"Training positives: {int(y_train.sum()):,}")
    print(f"Training negatives: {int((y_train == 0).sum()):,}")
    print(f"Positive-class weight: {positive_weight:.6f}")

    # Shuffle training batches, not validation batches.
    train_loader = create_dataloader(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = create_dataloader(
        X_val,
        y_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = create_baseline_model(
        input_size=len(SENSOR_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    trainer = BaselineTrainer(
        model=model,
        learning_rate=LEARNING_RATE,
        positive_class_weight=positive_weight,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training uses only training and validation loaders.
    # The official test set is not passed to the trainer.
    fit_result = trainer.fit(
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=EPOCHS,
        patience=PATIENCE,
        checkpoint_path=checkpoint_path,
    )

    # Save history if the trainer exposes a history attribute.
    history = getattr(trainer, "history", None)

    if history is not None:
        if isinstance(history, pd.DataFrame):
            history_df = history
        else:
            history_df = pd.DataFrame(history)

        if not history_df.empty:
            history_df.to_csv(history_path, index=False)
            print(f"Training history saved: {history_path}")
    else:
        print(
            "Note: trainer.history was not available; "
            "the checkpoint and training report will still be saved."
        )

    # Read the saved checkpoint to report the selected epoch and loss.
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Training finished but the expected checkpoint was not found. "
            "Check the BaselineTrainer.fit implementation."
        )

    try:
        saved = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        saved = torch.load(checkpoint_path, map_location="cpu")

    report = {
        "subset": subset,
        "status": "trained",
        "seed": SEED,
        "training_windows": int(len(X_train)),
        "validation_windows": int(len(X_val)),
        "training_positive_count": int(y_train.sum()),
        "training_negative_count": int((y_train == 0).sum()),
        "positive_class_weight": float(positive_weight),
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "early_stopping_patience": PATIENCE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "checkpoint": str(checkpoint_path),
        "selected_epoch": saved.get("epoch"),
        "best_validation_loss": saved.get("best_validation_loss"),
        "validation_metrics": saved.get("validation_metrics"),
        "fit_return_type": type(fit_result).__name__,
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)

    print("\nTraining completed.")
    print(f"Selected epoch: {report['selected_epoch']}")
    print(f"Best validation loss: {report['best_validation_loss']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Report: {report_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SensorGuard baselines for FD002-FD004."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if a subset checkpoint already exists.",
    )
    args = parser.parse_args()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("FD001 will not be retrained or modified.")

    reports = []

    for subset in SUBSETS:
        try:
            reports.append(train_subset(subset, force=args.force))
        except Exception as exc:
            print(f"\n[ERROR] {subset}: {exc}")
            raise

    summary_path = (
        PROJECT_ROOT
        / "results"
        / "metrics"
        / "additional_subset_training_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(reports, file, indent=2, default=str)

    print("\n" + "=" * 76)
    print("Additional-subset training run finished.")
    print(f"Summary: {summary_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
