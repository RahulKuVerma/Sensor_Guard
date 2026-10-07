import numpy as np

from src.evaluation.metrics import compute_classification_metrics


def test_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1])

    y_prob = np.array([
        0.05,
        0.20,
        0.40,
        0.60,
        0.80,
        0.95,
    ])

    metrics = compute_classification_metrics(
        y_true,
        y_prob,
        threshold=0.5,
    )

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert 0.0 <= metrics["auprc"] <= 1.0
    assert metrics["brier"] >= 0.0

    assert metrics["confusion_matrix"] == [
        [3, 0],
        [0, 3],
    ]

    print("Metrics test: PASS")


if __name__ == "__main__":
    test_metrics()