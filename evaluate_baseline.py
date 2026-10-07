import numpy as np
import torch

from src.data.loader import load_dataset
from src.features.dataset_builder import prepare_training_validation_test
from src.models.baseline import create_baseline_model
from src.models.dataset import create_dataloader
from src.evaluation.metrics import compute_classification_metrics


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

CHECKPOINT_PATH = "results/models/FD001/baseline_lstm.pt"


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 90)
print("SensorGuard - Official FD001 Test Evaluation")
print("=" * 90)

print(f"Device: {device}")


# ============================================================
# Load raw dataset
# ============================================================

dataset = load_dataset(SUBSET)

train_df = dataset["train"]
test_df = dataset["test"]
rul_df = dataset["rul"]

print()
print("Raw dataset:")
print(f"  Train rows : {len(train_df)}")
print(f"  Test rows  : {len(test_df)}")
print(f"  RUL rows   : {len(rul_df)}")


# ============================================================
# Recreate leakage-safe preprocessing
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

print()
print("Official test windows:")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_test shape: {y_test.shape}")


# ============================================================
# Create test DataLoader
# ============================================================

test_loader = create_dataloader(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
)

print(f"  Test batches: {len(test_loader)}")


# ============================================================
# Create model
# ============================================================

model = create_baseline_model(
    input_size=21,
    hidden_size=128,
    num_layers=2,
    dropout=0.30,
)

model = model.to(device)


# ============================================================
# Load best validation checkpoint
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

model.load_state_dict(checkpoint["model_state_dict"])

print()
print("Checkpoint loaded:")
print(f"  Path       : {CHECKPOINT_PATH}")
print(f"  Best epoch : {checkpoint.get('epoch', 'N/A')}")


if "val_loss" in checkpoint:
    print(f"  Val loss   : {checkpoint['val_loss']:.6f}")


# ============================================================
# Official test inference
# ============================================================

model.eval()

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


y_prob = np.concatenate(all_probabilities)
y_true = np.concatenate(all_targets).astype(int)


# ============================================================
# Sanity checks
# ============================================================

assert len(y_true) == len(y_prob)

assert np.isfinite(y_prob).all()

assert np.isin(y_true, [0, 1]).all()

print()
print("Inference completed:")
print(f"  Predictions: {len(y_prob)}")
print(f"  Probability min: {y_prob.min():.6f}")
print(f"  Probability max: {y_prob.max():.6f}")
print(f"  Probability mean: {y_prob.mean():.6f}")


# ============================================================
# Test class distribution
# ============================================================

normal_count = int((y_true == 0).sum())
early_fault_count = int((y_true == 1).sum())

print()
print("Official test class distribution:")
print(f"  Normal      : {normal_count}")
print(f"  Early fault : {early_fault_count}")
print(
    f"  Early fault ratio: "
    f"{early_fault_count / len(y_true):.4f}"
)


# ============================================================
# Compute metrics
# ============================================================

metrics = compute_classification_metrics(
    y_true=y_true,
    y_prob=y_prob,
    threshold=0.5,
)


# ============================================================
# Display results
# ============================================================

print()
print("=" * 90)
print("OFFICIAL FD001 TEST RESULTS")
print("=" * 90)

print(f"Threshold : {metrics['threshold']:.2f}")
print()

print(f"Accuracy  : {metrics['accuracy']:.6f}")
print(f"Precision : {metrics['precision']:.6f}")
print(f"Recall    : {metrics['recall']:.6f}")
print(f"F1        : {metrics['f1']:.6f}")
print(f"AUROC     : {metrics['auroc']:.6f}")
print(f"AUPRC     : {metrics['auprc']:.6f}")
print(f"Brier     : {metrics['brier']:.6f}")

print()
print("Confusion Matrix:")
print(
    np.array(
        metrics["confusion_matrix"]
    )
)

print()
print("=" * 90)
print("Official FD001 test evaluation completed.")
print("=" * 90)