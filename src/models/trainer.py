from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader


class BaselineTrainer:
    """
    Training engine for the SensorGuard clean baseline model.

    Responsibilities:
        - BCEWithLogitsLoss
        - Training-only positive-class weighting
        - Adam optimization
        - Training loop
        - Validation loop
        - Loss tracking
        - Accuracy / precision / recall / F1
        - Best-model checkpointing
        - Early stopping
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        positive_class_weight: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero."
            )

        self.model = model

        # -----------------------------------------------------
        # Device
        # -----------------------------------------------------

        if device is None:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device

        self.model.to(self.device)

        # -----------------------------------------------------
        # Positive-class weighting
        # -----------------------------------------------------

        if positive_class_weight is not None:

            if positive_class_weight <= 0:
                raise ValueError(
                    "positive_class_weight must be "
                    "greater than zero."
                )

            self.positive_class_weight = float(
                positive_class_weight
            )

            pos_weight = torch.tensor(
                [self.positive_class_weight],
                dtype=torch.float32,
                device=self.device,
            )

        else:

            self.positive_class_weight = None
            pos_weight = None

        # -----------------------------------------------------
        # Loss
        # -----------------------------------------------------

        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

        # -----------------------------------------------------
        # Optimizer
        # -----------------------------------------------------

        self.optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.learning_rate = learning_rate

        # -----------------------------------------------------
        # Training history
        # -----------------------------------------------------

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "validation_loss": [],
            "train_accuracy": [],
            "validation_accuracy": [],
            "train_precision": [],
            "validation_precision": [],
            "train_recall": [],
            "validation_recall": [],
            "train_f1": [],
            "validation_f1": [],
        }

        self.best_validation_loss = float(
            "inf"
        )

        self.best_epoch = None

    # =========================================================
    # Metrics
    # =========================================================

    @staticmethod
    def _classification_metrics(
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Calculate binary classification metrics.

        Predictions are generated using a probability
        threshold of 0.5.
        """

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= 0.5
        ).long()

        targets = targets.long()

        tp = int(
            (
                (predictions == 1)
                & (targets == 1)
            ).sum().item()
        )

        tn = int(
            (
                (predictions == 0)
                & (targets == 0)
            ).sum().item()
        )

        fp = int(
            (
                (predictions == 1)
                & (targets == 0)
            ).sum().item()
        )

        fn = int(
            (
                (predictions == 0)
                & (targets == 1)
            ).sum().item()
        )

        total = tp + tn + fp + fn

        accuracy = (
            (tp + tn) / total
            if total > 0
            else 0.0
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2.0 * precision * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # =========================================================
    # One training epoch
    # =========================================================

    def train_one_epoch(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:

        self.model.train()

        total_loss = 0.0
        total_samples = 0

        all_logits = []
        all_targets = []

        for X_batch, y_batch in dataloader:

            X_batch = X_batch.to(
                self.device
            )

            y_batch = y_batch.to(
                self.device
            )

            # -------------------------------------------------
            # Clear previous gradients
            # -------------------------------------------------

            self.optimizer.zero_grad()

            # -------------------------------------------------
            # Forward pass
            # -------------------------------------------------

            logits = self.model(
                X_batch
            )

            # -------------------------------------------------
            # Loss
            # -------------------------------------------------

            loss = self.criterion(
                logits,
                y_batch,
            )

            # -------------------------------------------------
            # Backpropagation
            # -------------------------------------------------

            loss.backward()

            # -------------------------------------------------
            # Parameter update
            # -------------------------------------------------

            self.optimizer.step()

            batch_size = X_batch.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            all_logits.append(
                logits.detach().cpu()
            )

            all_targets.append(
                y_batch.detach().cpu()
            )

        if total_samples == 0:
            raise RuntimeError(
                "Training DataLoader produced zero samples."
            )

        logits = torch.cat(
            all_logits
        )

        targets = torch.cat(
            all_targets
        )

        metrics = (
            self._classification_metrics(
                logits,
                targets,
            )
        )

        metrics["loss"] = (
            total_loss / total_samples
        )

        return metrics

    # =========================================================
    # Validation epoch
    # =========================================================

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:

        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        all_logits = []
        all_targets = []

        for X_batch, y_batch in dataloader:

            X_batch = X_batch.to(
                self.device
            )

            y_batch = y_batch.to(
                self.device
            )

            logits = self.model(
                X_batch
            )

            loss = self.criterion(
                logits,
                y_batch,
            )

            batch_size = X_batch.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            all_logits.append(
                logits.detach().cpu()
            )

            all_targets.append(
                y_batch.detach().cpu()
            )

        if total_samples == 0:
            raise RuntimeError(
                "Validation DataLoader produced zero samples."
            )

        logits = torch.cat(
            all_logits
        )

        targets = torch.cat(
            all_targets
        )

        metrics = (
            self._classification_metrics(
                logits,
                targets,
            )
        )

        metrics["loss"] = (
            total_loss / total_samples
        )

        return metrics

    # =========================================================
    # Training loop
    # =========================================================

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epochs: int = 50,
        patience: int = 8,
        checkpoint_path: Optional[
            str | Path
        ] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model with validation and early stopping.

        The best model is selected using validation loss.
        """

        if epochs <= 0:
            raise ValueError(
                "epochs must be greater than zero."
            )

        if patience <= 0:
            raise ValueError(
                "patience must be greater than zero."
            )

        if checkpoint_path is not None:

            checkpoint_path = Path(
                checkpoint_path
            )

            checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        epochs_without_improvement = 0

        for epoch in range(
            1,
            epochs + 1,
        ):

            train_metrics = (
                self.train_one_epoch(
                    train_loader
                )
            )

            validation_metrics = (
                self.validate(
                    validation_loader
                )
            )

            # -------------------------------------------------
            # Save history
            # -------------------------------------------------

            self.history[
                "train_loss"
            ].append(
                train_metrics["loss"]
            )

            self.history[
                "validation_loss"
            ].append(
                validation_metrics["loss"]
            )

            self.history[
                "train_accuracy"
            ].append(
                train_metrics["accuracy"]
            )

            self.history[
                "validation_accuracy"
            ].append(
                validation_metrics["accuracy"]
            )

            self.history[
                "train_precision"
            ].append(
                train_metrics["precision"]
            )

            self.history[
                "validation_precision"
            ].append(
                validation_metrics["precision"]
            )

            self.history[
                "train_recall"
            ].append(
                train_metrics["recall"]
            )

            self.history[
                "validation_recall"
            ].append(
                validation_metrics["recall"]
            )

            self.history[
                "train_f1"
            ].append(
                train_metrics["f1"]
            )

            self.history[
                "validation_f1"
            ].append(
                validation_metrics["f1"]
            )

            # -------------------------------------------------
            # Best validation model
            # -------------------------------------------------

            if (
                validation_metrics["loss"]
                < self.best_validation_loss
            ):

                self.best_validation_loss = (
                    validation_metrics["loss"]
                )

                self.best_epoch = epoch

                epochs_without_improvement = 0

                if checkpoint_path is not None:

                    self.save_checkpoint(
                        checkpoint_path,
                        epoch=epoch,
                        validation_metrics=(
                            validation_metrics
                        ),
                    )

            else:

                epochs_without_improvement += 1

            # -------------------------------------------------
            # Console output
            # -------------------------------------------------

            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Train Loss: "
                f"{train_metrics['loss']:.4f} | "
                f"Val Loss: "
                f"{validation_metrics['loss']:.4f} | "
                f"Train F1: "
                f"{train_metrics['f1']:.4f} | "
                f"Val F1: "
                f"{validation_metrics['f1']:.4f}"
            )

            # -------------------------------------------------
            # Early stopping
            # -------------------------------------------------

            if (
                epochs_without_improvement
                >= patience
            ):

                print(
                    f"Early stopping at epoch "
                    f"{epoch}."
                )

                break

        return self.history

    # =========================================================
    # Checkpoint
    # =========================================================

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        validation_metrics: Dict[str, float],
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": (
                self.model.state_dict()
            ),
            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),
            "best_validation_loss": (
                self.best_validation_loss
            ),
            "validation_metrics": (
                validation_metrics
            ),
            "learning_rate": (
                self.learning_rate
            ),
            "positive_class_weight": (
                self.positive_class_weight
            ),
        }

        torch.save(
            checkpoint,
            path,
        )

    # =========================================================
    # Load checkpoint
    # =========================================================

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> Dict:
        """
        Load a previously saved checkpoint.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        self.best_validation_loss = (
            checkpoint[
                "best_validation_loss"
            ]
        )

        self.best_epoch = checkpoint[
            "epoch"
        ]

        return checkpoint