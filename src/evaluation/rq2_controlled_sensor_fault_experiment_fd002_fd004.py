"""
SensorGuard — Controlled RQ2 Sensor-Fault Experiment

Purpose:
    Measure how synthetic sensor faults change early-fault predictions
    for engines with different ground-truth early-fault labels.

Design:
    - Uses the existing trained checkpoint for each C-MAPSS subset.
    - Fits preprocessing using the existing training-only pipeline.
    - Applies faults to selected sensor channels only.
    - Keeps labels and engine IDs unchanged.
    - Compares clean and degraded inputs on matched test windows.
    - Measures classification changes and MC-dropout uncertainty.
    - Does not retrain or tune on official test data.

Run from project root:
    python -m src.evaluation.rq2_controlled_sensor_fault_experiment
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.loader import load_dataset
from src.features.dataset_builder import (
    prepare_training_validation_test,
)
from src.features.windows import create_sliding_windows
from src.models.baseline import create_baseline_model
from src.sensors.degradation import SensorDegradationEngine
from src.uncertainty.mc_dropout import mc_dropout_predict


# ============================================================
# Configuration
# ============================================================

SUBSETS = ["FD002", "FD003", "FD004"]  # Debug run: remaining subsets

HORIZON = 30
WINDOW_SIZE = 30
STRIDE = 1

TRAIN_RATIO = 0.85
VALIDATION_RATIO = 0.15
RANDOM_SEED = 42

MC_PASSES = 10  # Debug run; restore to 50 for final experiments
BATCH_SIZE = 256
DECISION_THRESHOLD = 0.50

SENSOR_COLUMNS = [
    f"sensor_{i}" for i in range(1, 22)
]

# Select one channel to make the initial experiment easy to interpret.
# Later, repeat with additional individual channels and channel groups.
SELECTED_SENSORS = ["sensor_2"]

# Severity levels are dimensionless multiples of each sensor's
# standard deviation in the data supplied to the degradation method.
SEVERITIES = [0.10, 0.25, 0.50]

FAULT_TYPES = [
    "random_dropout",
    "gaussian_noise",
    "positive_bias",
    "positive_drift",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "rq2_controlled_sensor_fault_experiment"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Model loading
# ============================================================

def load_baseline(subset):
    """Load the existing checkpoint without retraining."""

    checkpoint_path = (
        PROJECT_ROOT
        / "results"
        / "models"
        / subset
        / "baseline_lstm.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}"
        )

    model = create_baseline_model(
        input_size=21,
        hidden_size=128,
        num_layers=2,
        dropout=0.30,
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
# Metrics
# ============================================================

def safe_auc(y_true, scores):
    """Return NaN when the labels contain only one class."""

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(roc_auc_score(y_true, scores))


def calculate_metrics(y_true, probabilities):
    """Calculate threshold and probability-quality metrics."""

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probabilities >= DECISION_THRESHOLD
    ).astype(int)

    return {
        "n_windows": int(len(y_true)),
        "positive_labels": int(y_true.sum()),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "auroc": safe_auc(y_true, probabilities),
        "auprc": (
            float(
                average_precision_score(
                    y_true,
                    probabilities,
                )
            )
            if y_true.sum() > 0
            else float("nan")
        ),
        "brier": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        "predicted_positive_rate": float(
            predictions.mean()
        ),
        "mean_probability": float(
            probabilities.mean()
        ),
    }


# ============================================================
# Apply one controlled sensor fault
# ============================================================

def apply_fault(
    clean_df,
    fault_type,
    severity,
    sensors,
    seed,
):
    """
    Return a degraded copy of the scaled test dataframe.

    The input dataframe contains engine_id, cycle, sensor columns,
    and the original early_fault labels.

    Missing scaled values are imputed with zero, corresponding to
    the training mean under StandardScaler.
    """

    engine = SensorDegradationEngine(
        random_seed=seed
    )

    result = clean_df.copy()

    if fault_type == "random_dropout":
        result = engine.random_dropout(
            result,
            sensors=sensors,
            dropout_rate=severity,
        )

    elif fault_type == "gaussian_noise":
        result = engine.add_noise(
            result,
            sensors=sensors,
            noise_std=severity,
        )

    elif fault_type == "positive_bias":
        result = engine.add_bias(
            result,
            sensors=sensors,
            bias_scale=severity,
            direction="positive",
        )

    elif fault_type == "positive_drift":
        result = engine.add_drift(
            result,
            sensors=sensors,
            drift_scale=severity,
            direction="positive",
        )

    else:
        raise ValueError(
            f"Unsupported fault type: {fault_type}"
        )

    result[SENSOR_COLUMNS] = (
        result[SENSOR_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    return result


# ============================================================
# Predict and calculate uncertainty
# ============================================================

def predict_windows(model, X):
    """Run MC dropout and return probabilities and uncertainty."""

    output = mc_dropout_predict(
        model=model,
        X=X,
        n_passes=MC_PASSES,
        batch_size=BATCH_SIZE,
        device=str(DEVICE),
    )

    return {
        "probability": np.asarray(
            output["mean_probability"]
        ).reshape(-1),
        "variance": np.asarray(
            output["predictive_variance"]
        ).reshape(-1),
        "mutual_information": np.asarray(
            output["mutual_information"]
        ).reshape(-1),
        "entropy": np.asarray(
            output["predictive_entropy"]
        ).reshape(-1),
    }


# ============================================================
# Run one subset
# ============================================================

def run_subset(subset):
    print(f"\n{'=' * 65}")
    print(f"RQ2 controlled sensor-fault experiment: {subset}")
    print(f"{'=' * 65}")

    dataset = load_dataset(subset)

    train_df = dataset["train"].copy()
    test_df = dataset["test"].copy()
    rul_df = dataset["rul"].copy()

    # The official test set stays separate from training and validation.
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
        random_seed=RANDOM_SEED,
    )

    model = load_baseline(subset)
    print("Preparing clean test predictions...", flush=True)

    # This dataframe has already been transformed with the scaler
    # fitted by the training-only pipeline.
    clean_df = (
        prepared["test_df"]
        .copy()
        .sort_values(["engine_id", "cycle"])
        .reset_index(drop=True)
    )

    y_clean = np.asarray(
        prepared["y_test"],
        dtype=int,
    )

    engine_ids_clean = np.asarray(
        prepared["test_engine_ids"]
    )

    X_clean = np.asarray(
        prepared["X_test"],
        dtype=np.float32,
    )

    if len(X_clean) != len(y_clean):
        raise RuntimeError(
            "Mismatch between clean windows and labels."
        )

    if len(X_clean) != len(engine_ids_clean):
        raise RuntimeError(
            "Mismatch between windows and engine IDs."
        )

    # Clean predictions are computed once and reused for every
    # paired comparison in this subset.
    print(
        f"Running clean MC Dropout on {len(X_clean):,} windows...",
        flush=True,
    )
    clean_prediction = predict_windows(
        model,
        X_clean,
    )

    clean_prob = clean_prediction["probability"]

    clean_class = (
        clean_prob >= DECISION_THRESHOLD
    ).astype(int)

    records = []
    prediction_frames = []

    # Record the clean condition.
    scenarios = [
        {
            "scenario": "clean",
            "fault_type": "none",
            "severity": 0.0,
            "sensors": "none",
            "seed": RANDOM_SEED,
            "prediction": clean_prediction,
            "X": X_clean,
            "df": clean_df,
        }
    ]

    # Apply each fault type and severity to the selected channel.
    for fault_type in FAULT_TYPES:
        for severity in SEVERITIES:
            seed = (
                RANDOM_SEED
                + sum(ord(ch) for ch in subset)
                + int(severity * 1000)
                + sum(ord(ch) for ch in fault_type)
            )

            print(
                f"Applying {fault_type}, severity={severity:.2f}...",
                flush=True,
            )
            degraded_df = apply_fault(
                clean_df=clean_df,
                fault_type=fault_type,
                severity=severity,
                sensors=SELECTED_SENSORS,
                seed=seed,
            )

            X_degraded, y_degraded, engine_ids = (
                create_sliding_windows(
                    degraded_df,
                    sensor_columns=SENSOR_COLUMNS,
                    window_size=WINDOW_SIZE,
                    stride=STRIDE,
                )
            )

            X_degraded = np.asarray(
                X_degraded,
                dtype=np.float32,
            )

            y_degraded = np.asarray(
                y_degraded,
                dtype=int,
            )

            engine_ids = np.asarray(engine_ids)

            # Confirm every scenario preserves the target labels
            # and engine ordering.
            if not np.array_equal(y_degraded, y_clean):
                raise RuntimeError(
                    f"Labels changed for {subset}, "
                    f"{fault_type}, severity={severity}."
                )

            if not np.array_equal(
                engine_ids,
                engine_ids_clean,
            ):
                raise RuntimeError(
                    f"Engine-window alignment changed for "
                    f"{subset}, {fault_type}, "
                    f"severity={severity}."
                )

            print(
                f"Running MC Dropout on {fault_type}, "
                f"severity={severity:.2f}...",
                flush=True,
            )
            prediction = predict_windows(
                model,
                X_degraded,
            )

            scenarios.append(
                {
                    "scenario": (
                        f"{fault_type}_s{severity:.2f}"
                    ),
                    "fault_type": fault_type,
                    "severity": severity,
                    "sensors": ",".join(
                        SELECTED_SENSORS
                    ),
                    "seed": seed,
                    "prediction": prediction,
                    "X": X_degraded,
                    "df": degraded_df,
                }
            )

    # --------------------------------------------------------
    # Summarize all scenarios
    # --------------------------------------------------------

    for item in scenarios:
        prediction = item["prediction"]
        probabilities = prediction["probability"]

        metrics = calculate_metrics(
            y_clean,
            probabilities,
        )

        current_class = (
            probabilities >= DECISION_THRESHOLD
        ).astype(int)

        row = {
            "subset": subset,
            "scenario": item["scenario"],
            "fault_type": item["fault_type"],
            "severity": item["severity"],
            "selected_sensors": item["sensors"],
            "random_seed": item["seed"],
            **metrics,
            "mean_predictive_variance": float(
                np.mean(prediction["variance"])
            ),
            "mean_mutual_information": float(
                np.mean(
                    prediction["mutual_information"]
                )
            ),
            "mean_predictive_entropy": float(
                np.mean(prediction["entropy"])
            ),
            "classification_change_rate": float(
                np.mean(current_class != clean_class)
            ),
            "mean_absolute_probability_shift": float(
                np.mean(
                    np.abs(probabilities - clean_prob)
                )
            ),
            "mean_signed_probability_shift": float(
                np.mean(probabilities - clean_prob)
            ),
        }

        # Measure effects separately for engines/windows whose
        # original label is early-fault vs not-early-fault.
        for label_value, label_name in [
            (0, "non_early_fault"),
            (1, "early_fault"),
        ]:
            mask = y_clean == label_value

            row[f"{label_name}_n"] = int(mask.sum())

            if mask.any():
                row[
                    f"{label_name}_mean_probability"
                ] = float(
                    probabilities[mask].mean()
                )

                row[
                    f"{label_name}_mean_variance"
                ] = float(
                    prediction["variance"][mask].mean()
                )

                row[
                    f"{label_name}_mean_mutual_information"
                ] = float(
                    prediction[
                        "mutual_information"
                    ][mask].mean()
                )

                row[
                    f"{label_name}_classification_change_rate"
                ] = float(
                    np.mean(
                        current_class[mask]
                        != clean_class[mask]
                    )
                )
            else:
                row[
                    f"{label_name}_mean_probability"
                ] = float("nan")

                row[
                    f"{label_name}_mean_variance"
                ] = float("nan")

                row[
                    f"{label_name}_mean_mutual_information"
                ] = float("nan")

                row[
                    f"{label_name}_classification_change_rate"
                ] = float("nan")

        records.append(row)

        # Save paired per-window predictions. This enables later
        # engine-clustered bootstrap and error analysis.
        prediction_frames.append(
            pd.DataFrame(
                {
                    "subset": subset,
                    "scenario": item["scenario"],
                    "fault_type": item["fault_type"],
                    "severity": item["severity"],
                    "selected_sensors": item["sensors"],
                    "engine_id": engine_ids_clean,
                    "y_true": y_clean,
                    "clean_probability": clean_prob,
                    "scenario_probability": probabilities,
                    "clean_prediction": clean_class,
                    "scenario_prediction": current_class,
                    "clean_variance": clean_prediction[
                        "variance"
                    ],
                    "scenario_variance": prediction[
                        "variance"
                    ],
                    "clean_mutual_information": (
                        clean_prediction[
                            "mutual_information"
                        ]
                    ),
                    "scenario_mutual_information": (
                        prediction[
                            "mutual_information"
                        ]
                    ),
                    "absolute_probability_shift": np.abs(
                        probabilities - clean_prob
                    ),
                    "classification_changed": (
                        current_class != clean_class
                    ),
                    "clean_error": (
                        clean_class != y_clean
                    ),
                    "scenario_error": (
                        current_class != y_clean
                    ),
                }
            )
        )

        print(
            f"{item['scenario']:<28} "
            f"F1={metrics['f1']:.4f} "
            f"AUROC={metrics['auroc']:.4f} "
            f"MI={row['mean_mutual_information']:.6f} "
            f"Changed={row['classification_change_rate']:.4f}"
        )

    subset_dir = OUTPUT_DIR / subset
    subset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df = pd.DataFrame(records)
    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    summary_path = (
        subset_dir / "rq2_scenario_summary.csv"
    )

    predictions_path = (
        subset_dir / "rq2_paired_predictions.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    report = {
        "subset": subset,
        "window_size": WINDOW_SIZE,
        "mc_passes": MC_PASSES,
        "decision_threshold": DECISION_THRESHOLD,
        "selected_sensors": SELECTED_SENSORS,
        "fault_types": FAULT_TYPES,
        "severity_levels": SEVERITIES,
        "n_test_windows": int(len(y_clean)),
        "n_test_engines": int(
            len(np.unique(engine_ids_clean))
        ),
        "positive_labels": int(y_clean.sum()),
        "negative_labels": int(
            len(y_clean) - y_clean.sum()
        ),
        "interpretation_limit": (
            "Synthetic sensor faults are controlled perturbations. "
            "Observed prediction changes and uncertainty do not "
            "establish causal attribution or constitute a dedicated "
            "sensor-health diagnosis."
        ),
        "summary_csv": str(summary_path),
        "paired_predictions_csv": str(
            predictions_path
        ),
    }

    with open(
        subset_dir / "rq2_run_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {predictions_path}")

    return summary_df


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_summaries = []

    for subset in SUBSETS:
        summary = run_subset(subset)
        all_summaries.append(summary)

    combined = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    combined_path = (
        OUTPUT_DIR / "rq2_all_subsets_summary.csv"
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    print("\n" + "=" * 65)
    print("RQ2 CONTROLLED EXPERIMENT COMPLETED")
    print("=" * 65)
    print(f"Combined summary: {combined_path}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
