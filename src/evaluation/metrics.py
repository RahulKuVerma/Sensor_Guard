from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """
    Compute classification metrics from binary targets
    and predicted probabilities.
    """

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array.")

    if y_prob.ndim != 1:
        raise ValueError("y_prob must be a 1D array.")

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, "
            f"y_prob={len(y_prob)}"
        )

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob contains NaN or infinite values.")

    if not np.isfinite(y_true).all():
        raise ValueError("y_true contains NaN or infinite values.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only binary values: 0 or 1.")

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "auroc": float(
            roc_auc_score(y_true, y_prob)
        ),
        "auprc": float(
            average_precision_score(y_true, y_prob)
        ),
        "brier": float(
            brier_score_loss(y_true, y_prob)
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).tolist(),
        "threshold": float(threshold),
    }

    return metrics