
"""
SensorGuard: Terminal-Window Evaluation

Evaluates one final 30-cycle window per official C-MAPSS test engine.
Short trajectories are left-padded by repeating their first observed row.

This is a binary early-fault classification evaluation.
It does not retrain models or modify existing results.

Run from the project root:
    python -m src.evaluation.terminal_window_evaluation
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import load_dataset
from src.features.dataset_builder import (
    prepare_training_validation_test,
)
from src.models.baseline import create_baseline_model


# ============================================================
# Configuration
# ============================================================

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

HORIZON = 30
WINDOW_SIZE = 30

TRAIN_RATIO = 0.85
VALIDATION_RATIO = 0.15
RANDOM_SEED = 42

BATCH_SIZE = 256
THRESHOLD = 0.50

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.30

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUMMARY_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "terminal_window_evaluation_padded"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load an existing checkpoint
# ============================================================

def load_baseline(checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = create_baseline_model(
        input_size=len(SENSOR_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# Predict probabilities
# ============================================================

@torch.no_grad()
def predict_probabilities(model, X):
    if len(X) == 0:
        raise ValueError("No terminal windows were provided.")

    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    predictions = []

    model.eval()

    for (X_batch,) in loader:
        X_batch = X_batch.to(DEVICE)

        logits = model(X_batch)
        probabilities = torch.sigmoid(logits)

        predictions.extend(
            probabilities.detach().cpu().numpy().tolist()
        )

    return np.asarray(predictions, dtype=float)


# ============================================================
# Classification metrics
# ============================================================

def calculate_metrics(y_true, probabilities):
    predictions = (
        probabilities >= THRESHOLD
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "samples": int(len(y_true)),
        "positive_labels": int(np.sum(y_true == 1)),
        "negative_labels": int(np.sum(y_true == 0)),
        "threshold": THRESHOLD,
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "auroc": (
            float(roc_auc_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2
            else None
        ),
        "auprc": (
            float(
                average_precision_score(
                    y_true, probabilities
                )
            )
            if np.any(y_true == 1)
            else None
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "log_loss": float(
            log_loss(
                y_true,
                np.clip(probabilities, 1e-7, 1 - 1e-7),
                labels=[0, 1],
            )
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    return metrics, predictions


# ============================================================
# Confusion-matrix plot
# ============================================================

def save_confusion_matrix(metrics, output_path):
    matrix = np.array([
        [
            metrics["true_negative"],
            metrics["false_positive"],
        ],
        [
            metrics["false_negative"],
            metrics["true_positive"],
        ],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(matrix)

    ax.set_title("Terminal-Window Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal", "Early fault"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Early fault"])

    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                fontsize=12,
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ============================================================
# Evaluate one subset
# ============================================================

def evaluate_subset(subset):
    print("\n" + "=" * 72)
    print(f"Terminal-window evaluation: {subset}")
    print("=" * 72)

    checkpoint_path = (
        PROJECT_ROOT
        / "results"
        / "models"
        / subset
        / "baseline_lstm.pt"
    )

    output_dir = SUMMARY_DIR / subset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original dataset.
    dataset = load_dataset(subset)

    train_df = dataset["train"].copy()
    test_df = dataset["test"].copy()
    rul_df = dataset["rul"].copy()

    train_df = (
        train_df.sort_values(
            ["engine_id", "cycle"], kind="stable"
        ).reset_index(drop=True)
    )

    test_df = (
        test_df.sort_values(
            ["engine_id", "cycle"], kind="stable"
        ).reset_index(drop=True)
    )

    # Reconstruct the existing training-only scaler.
    # This function also reconstructs trajectory windows,
    # but we will not use those windows for this evaluation.
    prepared = prepare_training_validation_test(
        train_df=train_df,
        test_df=test_df,
        rul_df=rul_df,
        sensor_columns=SENSOR_COLUMNS,
        horizon=HORIZON,
        train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO,
        window_size=WINDOW_SIZE,
        stride=1,
        random_seed=RANDOM_SEED,
    )

    # Already transformed using a scaler fitted on the
    # original training partition only.
    scaled_test = prepared["test_df"]

    # Map official terminal RUL values to sorted test engine IDs.
    engine_ids = sorted(
        test_df["engine_id"].unique().tolist()
    )

    terminal_rul_values = rul_df["rul"].to_numpy(
        dtype=np.float64
    )

    if len(engine_ids) != len(terminal_rul_values):
        raise ValueError(
            f"{subset}: test engine count ({len(engine_ids)}) "
            f"does not match official RUL count "
            f"({len(terminal_rul_values)})."
        )

    terminal_rul_map = dict(
        zip(engine_ids, terminal_rul_values)
    )

    X_terminal = []
    records = []
    skipped_engines = []

    # Create exactly one final window for every test engine.
    # For trajectories shorter than WINDOW_SIZE, left-pad by
    # repeating the first observed scaled sensor row. This is
    # synthetic padding and is recorded in the prediction file.
    for engine_id, engine_df in scaled_test.groupby(
        "engine_id", sort=True
    ):
        engine_df = (
            engine_df.sort_values("cycle")
            .reset_index(drop=True)
        )

        sensor_values = engine_df[
            SENSOR_COLUMNS
        ].to_numpy(dtype=np.float32)

        observed_length = len(sensor_values)
        if observed_length == 0:
            skipped_engines.append(int(engine_id))
            continue

        if not np.isfinite(sensor_values).all():
            raise ValueError(
                f"{subset}, engine {engine_id}: "
                "sensor data contains NaN or infinity."
            )

        padding_rows = max(0, WINDOW_SIZE - observed_length)

        if padding_rows > 0:
            left_padding = np.repeat(
                sensor_values[:1], padding_rows, axis=0
            )
            terminal_window = np.concatenate(
                [left_padding, sensor_values], axis=0
            )
        else:
            terminal_window = sensor_values[-WINDOW_SIZE:]

        if terminal_window.shape != (
            WINDOW_SIZE, len(SENSOR_COLUMNS)
        ):
            raise RuntimeError(
                f"Invalid window shape for {subset}, engine "
                f"{engine_id}: {terminal_window.shape}"
            )

        terminal_rul = float(terminal_rul_map[int(engine_id)])
        target = int(terminal_rul <= HORIZON)
        X_terminal.append(terminal_window)

        records.append({
            "subset": subset,
            "engine_id": int(engine_id),
            "final_observed_cycle": int(engine_df["cycle"].iloc[-1]),
            "observed_cycles": observed_length,
            "window_size": WINDOW_SIZE,
            "padding_rows": padding_rows,
            "padding_method": (
                "repeat_first_observation" if padding_rows > 0 else "none"
            ),
            "official_terminal_rul": terminal_rul,
            "true_early_fault": target,
        })

    if not X_terminal:
        raise ValueError(f"{subset}: no terminal windows were created.")

    X_terminal = np.stack(X_terminal).astype(np.float32)
    y_true = np.asarray(
        [record["true_early_fault"] for record in records],
        dtype=np.int64,
    )

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test engines in dataset: {len(engine_ids)}")
    print(f"Evaluated terminal windows: {len(X_terminal)}")
    print(f"Padded engines: {sum(r['padding_rows'] > 0 for r in records)}")
    print(f"Skipped engines (empty trajectories): {len(skipped_engines)}")

    # Predict with existing checkpoint.
    model = load_baseline(checkpoint_path)

    probabilities = predict_probabilities(
        model, X_terminal
    )

    metrics, predictions = calculate_metrics(
        y_true, probabilities
    )

    # Save per-engine predictions.
    prediction_df = pd.DataFrame(records)
    prediction_df["predicted_probability"] = probabilities
    prediction_df["predicted_early_fault"] = predictions
    prediction_df["correct"] = (
        prediction_df["true_early_fault"]
        == prediction_df["predicted_early_fault"]
    )

    prediction_path = (
        output_dir / "terminal_predictions.csv"
    )
    prediction_df.to_csv(prediction_path, index=False)

    # Save metrics.
    report = {
        "subset": subset,
        "evaluation_protocol": (
            "one final 30-cycle window per test engine; short trajectories left-padded by repeating the first observed scaled sensor row"
        ),
        "target_definition": (
            f"early_fault = 1 if official terminal RUL <= {HORIZON}"
        ),
        "window_size": WINDOW_SIZE,
        "horizon": HORIZON,
        "threshold": THRESHOLD,
        "test_engines_in_dataset": len(engine_ids),
        "evaluated_engines": len(X_terminal),
        "padded_engines": int(sum(r["padding_rows"] > 0 for r in records)),
        "padding_method": "repeat_first_observation_left_padding",
        "skipped_engines": skipped_engines,
        "device": str(DEVICE),
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
    }

    report_path = output_dir / "terminal_metrics.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    pd.DataFrame([{
        "subset": subset,
        **metrics,
    }]).to_csv(
        output_dir / "terminal_metrics.csv",
        index=False,
    )

    save_confusion_matrix(
        metrics,
        output_dir / "confusion_matrix.png",
    )

    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")

    print(f"\nPredictions: {prediction_path}")
    print(f"Metrics JSON: {report_path}")

    return {
        "subset": subset,
        "test_engines_in_dataset": len(engine_ids),
        "evaluated_engines": len(X_terminal),
        "padded_engines": int(sum(r["padding_rows"] > 0 for r in records)),
        "skipped_engines": len(skipped_engines),
        **metrics,
    }


# ============================================================
# Main
# ============================================================

def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SensorGuard - Terminal-Window Evaluation")
    print("=" * 72)
    print("No model retraining will be performed.")
    print("Short trajectories: left-pad with first observed scaled row.")
    print(f"Output directory: {SUMMARY_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Early-fault horizon: {HORIZON}")

    results = []
    failures = []

    for subset in SUBSETS:
        try:
            results.append(evaluate_subset(subset))
        except Exception as exc:
            failures.append({
                "subset": subset,
                "error": str(exc),
            })
            print(f"\nERROR evaluating {subset}: {exc}")

    if results:
        summary_df = pd.DataFrame(results)

        summary_csv = (
            SUMMARY_DIR / "terminal_evaluation_summary.csv"
        )
        summary_df.to_csv(summary_csv, index=False)

        summary_json = (
            SUMMARY_DIR / "terminal_evaluation_summary.json"
        )
        with summary_json.open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                results,
                file,
                indent=2,
                allow_nan=False,
            )

        print("\n" + "=" * 72)
        print("TERMINAL-WINDOW SUMMARY")
        print("=" * 72)

        display_columns = [
            "subset",
            "evaluated_engines",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auroc",
            "auprc",
            "brier_score",
            "log_loss",
        ]

        print(
            summary_df[display_columns].to_string(
                index=False
            )
        )

        print(f"\nSummary CSV: {summary_csv}")
        print(f"Summary JSON: {summary_json}")

    if failures:
        failure_path = SUMMARY_DIR / "evaluation_errors.json"

        with failure_path.open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(failures, file, indent=2)

        print("\nSome subsets could not be evaluated:")
        for failure in failures:
            print(
                f"  {failure['subset']}: {failure['error']}"
            )

        print(f"Errors saved to: {failure_path}")

        raise RuntimeError(
            f"{len(failures)} subset(s) failed. "
            "Review the errors before using the summary."
        )

    print("\nTerminal-window evaluation completed.")


if __name__ == "__main__":
    main()
