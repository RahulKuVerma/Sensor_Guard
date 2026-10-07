import tempfile
from pathlib import Path

import numpy as np
import torch

from src.models.baseline import (
    create_baseline_model,
)

from src.models.dataset import (
    create_dataloader,
)

from src.models.trainer import (
    BaselineTrainer,
)


def create_dummy_data():
    """
    Create a small deterministic dataset for
    testing the training engine.
    """

    rng = np.random.default_rng(
        seed=42
    )

    X = rng.normal(
        0.0,
        1.0,
        size=(128, 30, 21),
    ).astype(
        np.float32
    )

    # Balanced binary target.
    y = np.array(
        [0, 1] * 64,
        dtype=np.int64,
    )

    return X, y


def test_trainer_creation():

    model = create_baseline_model()

    trainer = BaselineTrainer(
        model=model,
        learning_rate=0.001,
        positive_class_weight=1.0,
    )

    assert trainer.model is model

    assert trainer.learning_rate == 0.001

    assert (
        trainer.positive_class_weight
        == 1.0
    )

    assert trainer.criterion is not None

    assert trainer.optimizer is not None

    print(
        "Trainer creation: PASS"
    )


def test_classification_metrics():

    logits = torch.tensor(
        [
            10.0,
            10.0,
            -10.0,
            -10.0,
        ]
    )

    targets = torch.tensor(
        [
            1.0,
            0.0,
            0.0,
            1.0,
        ]
    )

    metrics = (
        BaselineTrainer._classification_metrics(
            logits,
            targets,
        )
    )

    assert metrics["accuracy"] == 0.5

    assert metrics["precision"] == 0.5

    assert metrics["recall"] == 0.5

    assert metrics["f1"] == 0.5

    print(
        "Classification metrics: PASS"
    )


def test_train_one_epoch():

    X, y = create_dummy_data()

    loader = create_dataloader(
        X=X,
        y=y,
        batch_size=32,
        shuffle=True,
    )

    model = create_baseline_model()

    trainer = BaselineTrainer(
        model=model,
        learning_rate=0.001,
        positive_class_weight=1.0,
    )

    metrics = trainer.train_one_epoch(
        loader
    )

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics

    assert np.isfinite(
        metrics["loss"]
    )

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0

    print(
        "Training epoch: PASS"
    )

    print(
        f"Training loss: "
        f"{metrics['loss']:.6f}"
    )


def test_validation():

    X, y = create_dummy_data()

    loader = create_dataloader(
        X=X,
        y=y,
        batch_size=32,
        shuffle=False,
    )

    model = create_baseline_model()

    trainer = BaselineTrainer(
        model=model,
        learning_rate=0.001,
        positive_class_weight=1.0,
    )

    metrics = trainer.validate(
        loader
    )

    assert np.isfinite(
        metrics["loss"]
    )

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0

    print(
        "Validation epoch: PASS"
    )

    print(
        f"Validation loss: "
        f"{metrics['loss']:.6f}"
    )


def test_fit_and_checkpoint():

    X, y = create_dummy_data()

    train_loader = create_dataloader(
        X=X[:96],
        y=y[:96],
        batch_size=32,
        shuffle=True,
    )

    validation_loader = create_dataloader(
        X=X[96:],
        y=y[96:],
        batch_size=32,
        shuffle=False,
    )

    model = create_baseline_model()

    trainer = BaselineTrainer(
        model=model,
        learning_rate=0.001,
        positive_class_weight=1.0,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:

        checkpoint_path = (
            Path(tmp_dir)
            / "baseline_test.pt"
        )

        history = trainer.fit(
            train_loader=train_loader,
            validation_loader=validation_loader,
            epochs=3,
            patience=3,
            checkpoint_path=checkpoint_path,
        )

        assert len(
            history["train_loss"]
        ) > 0

        assert len(
            history["validation_loss"]
        ) > 0

        assert checkpoint_path.exists()

        print(
            "Training loop: PASS"
        )

        print(
            f"Best epoch: "
            f"{trainer.best_epoch}"
        )

        # -----------------------------------------------------
        # Test checkpoint loading.
        # -----------------------------------------------------

        new_model = create_baseline_model()

        new_trainer = BaselineTrainer(
            model=new_model,
            learning_rate=0.001,
            positive_class_weight=1.0,
        )

        checkpoint = (
            new_trainer.load_checkpoint(
                checkpoint_path
            )
        )

        assert (
            checkpoint["epoch"]
            == new_trainer.best_epoch
        )

        print(
            "Checkpoint save/load: PASS"
        )


def test_class_weight():

    model = create_baseline_model()

    trainer = BaselineTrainer(
        model=model,
        learning_rate=0.001,
        positive_class_weight=3.5,
    )

    assert (
        trainer.positive_class_weight
        == 3.5
    )

    loss_pos_weight = (
        trainer.criterion.pos_weight
    )

    assert loss_pos_weight is not None

    assert (
        loss_pos_weight.item()
        == 3.5
    )

    print(
        "Positive-class weighting: PASS"
    )


if __name__ == "__main__":

    print("=" * 80)

    print(
        "SensorGuard - Training Engine Tests"
    )

    print("=" * 80)

    test_trainer_creation()

    test_classification_metrics()

    test_train_one_epoch()

    test_validation()

    test_fit_and_checkpoint()

    test_class_weight()

    print("=" * 80)

    print(
        "ALL TRAINING ENGINE TESTS PASSED"
    )

    print("=" * 80)