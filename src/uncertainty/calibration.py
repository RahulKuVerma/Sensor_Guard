
"""
SensorGuard - Probability calibration utilities.

These functions evaluate calibration; they do not fit a calibrator
or change the model's predictions.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


def _validate_inputs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary labels and probability predictions."""

    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)

    if len(y_true) == 0:
        raise ValueError("Inputs must contain at least one sample.")

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have equal lengths.")

    if not np.isfinite(y_true).all():
        raise ValueError("y_true contains non-finite values.")

    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob contains non-finite values.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1.")

    if ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("y_prob must contain probabilities in [0, 1].")

    return y_true.astype(int), y_prob


def calibration_bin_statistics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> list[dict]:
    """
    Calculate equal-width-bin calibration statistics.

    Empty bins are omitted from the returned list.
    The last bin includes probability 1.0.
    """

    y_true, y_prob = _validate_inputs(y_true, y_prob)

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, edges[1:-1], right=False)

    results = []

    for bin_index in range(n_bins):
        mask = bin_ids == bin_index

        if not mask.any():
            continue

        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())

        results.append(
            {
                "bin": int(bin_index),
                "bin_lower": float(edges[bin_index]),
                "bin_upper": float(edges[bin_index + 1]),
                "count": int(mask.sum()),
                "mean_confidence": confidence,
                "observed_frequency": accuracy,
                "absolute_calibration_error": abs(
                    confidence - accuracy
                ),
            }
        )

    return results


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Calculate ECE using equal-width probability bins.

    ECE = sum_b (n_b / N) * |accuracy_b - confidence_b|.
    Lower values indicate smaller average calibration gaps.
    """

    y_true, y_prob = _validate_inputs(y_true, y_prob)

    bins = calibration_bin_statistics(
        y_true,
        y_prob,
        n_bins=n_bins,
    )

    n_samples = len(y_true)

    return float(
        sum(
            item["count"] / n_samples
            * item["absolute_calibration_error"]
            for item in bins
        )
    )


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Return the largest absolute calibration gap among nonempty bins."""

    bins = calibration_bin_statistics(
        y_true,
        y_prob,
        n_bins=n_bins,
    )

    if not bins:
        return float("nan")

    return float(
        max(
            item["absolute_calibration_error"]
            for item in bins
        )
    )


def calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Calculate a set of binary probability calibration metrics."""

    y_true, y_prob = _validate_inputs(y_true, y_prob)

    # Clipping avoids undefined infinite log loss at exact 0 or 1.
    clipped_prob = np.clip(y_prob, 1e-7, 1.0 - 1e-7)

    bins = calibration_bin_statistics(
        y_true,
        y_prob,
        n_bins=n_bins,
    )

    return {
        "samples": int(len(y_true)),
        "ece": expected_calibration_error(
            y_true, y_prob, n_bins=n_bins
        ),
        "mce": maximum_calibration_error(
            y_true, y_prob, n_bins=n_bins
        ),
        "brier": float(
            brier_score_loss(y_true, y_prob)
        ),
        "log_loss": float(
            log_loss(
                y_true,
                clipped_prob,
                labels=[0, 1],
            )
        ),
        "n_bins": int(n_bins),
        "nonempty_bins": int(len(bins)),
        "calibration_bins": bins,
    }
