import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from src.data.loader import load_dataset
from src.evaluation.metrics import compute_classification_metrics
from src.features.dataset_builder import prepare_training_validation_test
from src.models.baseline import create_baseline_model
from src.models.dataset import create_dataloader


# ============================================================
# Configuration
# ============================================================

SUBSET = "FD001"

HORIZON = 30
TRAIN_RATIO = 0.85
VALIDATION_RATIO = 0.15
WINDOW_SIZE = 30
STRIDE = 1
RANDOM_SEED = 42

BATCH_SIZE = 64

CHECKPOINT_PATH = Path(
    "results/models/FD001/baseline_lstm.pt"
)

METRICS_DIR = Path(
    "results/metrics/FD001"
)

FIGURES_DIR = Path(
    "results/figures/FD001"
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 90)
print("SensorGuard - Baseline Evaluation Artifacts")
print("=" * 90)

print(f"Device: {device}")


# ============================================================
# Load dataset
# ============================================================

dataset = load_dataset(SUBSET)

train_df = dataset["train"]
test_df = dataset["test"]
rul_df = dataset["rul"]


# ============================================================
# Prepare test dataset
# ============================================================

prepared = prepare_training_validation_test(
    train_df=train_df,
    test_df=test_df,
    rul_df=rul_df,
    horizon=HORIZON,
    train_ratio=TRAIN_RATIO,
    validation_ratio=VALIDATION_RATIO,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    random_seed=RANDOM_SEED,
)

X_test = prepared["X_test"]
y_test = prepared["y_test"]


test_loader = create_dataloader(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
)


# ============================================================
# Create and load model
# ============================================================

model = create_baseline_model(
    input_size=21,
    hidden_size=128,
    num_layers=2,
    dropout=0.30,
)

model = model.to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# Inference
# ============================================================

all_probabilities = []
all_targets = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)

        logits = model(X_batch)

        probabilities = torch.sigmoid(logits)

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        all_targets.append(
            y_batch.numpy()
        )


y_prob = np.concatenate(
    all_probabilities
)

y_true = np.concatenate(
    all_targets
).astype(int)


# ============================================================
# Metrics
# ============================================================

metrics = compute_classification_metrics(
    y_true=y_true,
    y_prob=y_prob,
    threshold=0.5,
)


# ============================================================
# Save metrics JSON
# ============================================================

metrics_for_json = {
    key: value
    for key, value in metrics.items()
}

metrics_for_json.update(
    {
        "dataset": "NASA C-MAPSS FD001",
        "subset": SUBSET,
        "model": "LSTM Baseline",
        "checkpoint": str(CHECKPOINT_PATH),
        "horizon": HORIZON,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "threshold": 0.5,
        "test_samples": int(len(y_true)),
        "normal_samples": int((y_true == 0).sum()),
        "early_fault_samples": int((y_true == 1).sum()),
    }
)

metrics_path = (
    METRICS_DIR /
    "baseline_test_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics_for_json,
        f,
        indent=4,
    )


# ============================================================
# Save predictions
# ============================================================

predictions_path = (
    METRICS_DIR /
    "baseline_test_predictions.csv"
)

y_pred = (
    y_prob >= 0.5
).astype(int)

prediction_data = np.column_stack(
    [
        np.arange(len(y_true)),
        y_true,
        y_prob,
        y_pred,
    ]
)

np.savetxt(
    predictions_path,
    prediction_data,
    delimiter=",",
    header="sample_index,y_true,y_probability,y_pred",
    comments="",
    fmt=[
        "%d",
        "%d",
        "%.8f",
        "%d",
    ],
)


# ============================================================
# Confusion Matrix
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

ConfusionMatrixDisplay(
    confusion_matrix=np.array(
        metrics["confusion_matrix"]
    ),
    display_labels=[
        "Normal",
        "Early Fault",
    ],
).plot(
    ax=ax,
    values_format="d",
)

ax.set_title(
    "SensorGuard Baseline - FD001 Test Confusion Matrix"
)

fig.tight_layout()

confusion_path = (
    FIGURES_DIR /
    "baseline_confusion_matrix.png"
)

fig.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# ROC Curve
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

RocCurveDisplay.from_predictions(
    y_true,
    y_prob,
    ax=ax,
)

ax.set_title(
    "SensorGuard Baseline - FD001 ROC Curve"
)

fig.tight_layout()

roc_path = (
    FIGURES_DIR /
    "baseline_roc_curve.png"
)

fig.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Precision-Recall Curve
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

PrecisionRecallDisplay.from_predictions(
    y_true,
    y_prob,
    ax=ax,
)

ax.set_title(
    "SensorGuard Baseline - FD001 Precision-Recall Curve"
)

fig.tight_layout()

pr_path = (
    FIGURES_DIR /
    "baseline_precision_recall_curve.png"
)

fig.savefig(
    pr_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Final report
# ============================================================

print()
print("=" * 90)
print("BASELINE ARTIFACTS CREATED")
print("=" * 90)

print()
print("Metrics:")
print(f"  {metrics_path}")

print()
print("Predictions:")
print(f"  {predictions_path}")

print()
print("Figures:")

print(f"  {confusion_path}")
print(f"  {roc_path}")
print(f"  {pr_path}")

print()
print("Baseline metrics:")
print(f"  Accuracy  : {metrics['accuracy']:.6f}")
print(f"  Precision : {metrics['precision']:.6f}")
print(f"  Recall    : {metrics['recall']:.6f}")
print(f"  F1        : {metrics['f1']:.6f}")
print(f"  AUROC     : {metrics['auroc']:.6f}")
print(f"  AUPRC     : {metrics['auprc']:.6f}")
print(f"  Brier     : {metrics['brier']:.6f}")

print()
print("=" * 90)
print("Step 7 completed.")
print("=" * 90)