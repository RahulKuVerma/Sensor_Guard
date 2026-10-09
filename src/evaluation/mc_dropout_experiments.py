
"""
SensorGuard - MC-Dropout Uncertainty Experiments

Evaluates the trained FD001 LSTM on clean and synthetically
degraded sensor data using Monte Carlo Dropout.

Run from the project root:
    python -m src.evaluation.mc_dropout_experiments

Important:
- Does not retrain the model.
- Does not modify raw C-MAPSS files.
- Fits the scaler on the original training partition only.
- Applies degradation independently to copies of official test data.
- Uses identical labels and window ordering across scenarios.
- Imputes missing scaled sensor values with 0.0.
- Does not tune the model or decision threshold on test data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.data.loader import load_dataset
from src.data.test_targets import create_test_targets
from src.features.dataset_builder import (
    prepare_training_validation_test,
)
from src.features.windows import create_sliding_windows
from src.models.baseline import create_baseline_model
from src.evaluation.metrics import compute_classification_metrics
from src.sensors.degradation import SensorDegradationEngine
from src.uncertainty.mc_dropout import mc_dropout_predict


# ============================================================
# Configuration
# ============================================================

SUBSET = "FD001"

HORIZON = 30
WINDOW_SIZE = 30
STRIDE = 1

TRAIN_RATIO = 0.85
VALIDATION_RATIO = 0.15
RANDOM_SEED = 42

BATCH_SIZE = 256
N_MC_PASSES = 50
THRESHOLD = 0.50

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / SUBSET
    / "baseline_lstm.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / SUBSET
    / "mc_dropout_experiments"
)

SENSOR_COLUMNS = [
    f"sensor_{i}" for i in range(1, 22)
]

SCENARIOS = [
    "clean",
    "random_dropout",
    "missing_interval",
    "gaussian_noise",
    "positive_bias",
    "positive_drift",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load the existing trained baseline
# ============================================================

def load_baseline(checkpoint_path: Path):
    """Load the existing trained model without retraining."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {checkpoint_path}"
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
        # Compatibility with PyTorch versions without weights_only.
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
# Construct independent degradation scenarios
# ============================================================

def build_scenarios(
    official_test: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Generate six independent scenarios from the same test data."""

    engine = SensorDegradationEngine(
        random_seed=RANDOM_SEED
    )

    scenarios = {
        "clean": official_test.copy(deep=True),

        "random_dropout": engine.random_dropout(
            official_test,
            sensors=SENSOR_COLUMNS,
            dropout_rate=0.10,
        ),

        "missing_interval": engine.missing_interval(
            official_test,
            sensors=SENSOR_COLUMNS,
            interval_length=10,
            num_intervals=1,
        ),

        "gaussian_noise": engine.add_noise(
            official_test,
            sensors=SENSOR_COLUMNS,
            noise_std=0.20,
        ),

        "positive_bias": engine.add_bias(
            official_test,
            sensors=SENSOR_COLUMNS,
            bias_scale=0.20,
            direction="positive",
        ),

        "positive_drift": engine.add_drift(
            official_test,
            sensors=SENSOR_COLUMNS,
            drift_scale=0.50,
            direction="positive",
        ),
    }

    return scenarios


# ============================================================
# Scale data and create windows
# ============================================================

def prepare_scenario_windows(
    scenario_df: pd.DataFrame,
    scaler,
):
    """
    Scale using the training-fitted scaler and create windows.

    Missing values are imputed with zero after scaling.
    For StandardScaler, zero represents the fitted training mean.
    """

    scaled = scaler.transform(scenario_df)

    sensor_values = scaled[SENSOR_COLUMNS].to_numpy(
        dtype=np.float64
    )

    if np.isinf(sensor_values).any():
        raise ValueError(
            "Infinite sensor values found after scaling."
        )

    missing_mask = np.isnan(sensor_values)

    scaled.loc[:, SENSOR_COLUMNS] = (
        scaled[SENSOR_COLUMNS].fillna(0.0)
    )

    X, y, engine_ids = create_sliding_windows(
        scaled,
        sensor_columns=SENSOR_COLUMNS,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    if len(X) == 0:
        raise ValueError(
            "No windows were generated. Check trajectory lengths."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Model windows contain NaN or infinite values."
        )

    return (
        X,
        np.asarray(y).astype(int),
        np.asarray(engine_ids),
        int(missing_mask.sum()),
    )


# ============================================================
# Evaluate uncertainty against prediction errors
# ============================================================

def safe_error_detection_auroc(
    y_error: np.ndarray,
    uncertainty: np.ndarray,
) -> float:
    """
    Measure how well uncertainty ranks incorrect predictions.

    Incorrect predictions are the positive class.
    Returns NaN if there are no errors or no correct predictions.
    """

    y_error = np.asarray(y_error).astype(int)
    uncertainty = np.asarray(uncertainty, dtype=float)

    if len(np.unique(y_error)) < 2:
        return float("nan")

    return float(
        roc_auc_score(y_error, uncertainty)
    )


def summarize_scenario(
    y_true: np.ndarray,
    mean_probability: np.ndarray,
    uncertainty: dict[str, np.ndarray],
) -> dict:
    """Calculate classification and uncertainty summary metrics."""

    classification = compute_classification_metrics(
        y_true=y_true,
        y_prob=mean_probability,
        threshold=THRESHOLD,
    )

    y_pred = (
        mean_probability >= THRESHOLD
    ).astype(int)

    errors = (y_pred != y_true).astype(int)
    correct_mask = errors == 0
    error_mask = errors == 1

    mi = uncertainty["mutual_information"]
    variance = uncertainty["predictive_variance"]
    entropy = uncertainty["predictive_entropy"]
    expected_entropy = uncertainty["expected_entropy"]

    return {
        "samples": int(len(y_true)),
        "accuracy": float(classification["accuracy"]),
        "precision": float(classification["precision"]),
        "recall": float(classification["recall"]),
        "f1": float(classification["f1"]),
        "auroc": float(classification["auroc"]),
        "auprc": float(classification["auprc"]),
        "brier": float(classification["brier"]),
        "threshold": float(classification["threshold"]),
        "confusion_matrix": classification["confusion_matrix"],

        "error_count": int(errors.sum()),
        "error_rate": float(errors.mean()),

        "mean_predictive_variance": float(variance.mean()),
        "median_predictive_variance": float(np.median(variance)),

        "mean_predictive_entropy": float(entropy.mean()),
        "mean_expected_entropy": float(expected_entropy.mean()),
        "mean_mutual_information": float(mi.mean()),
        "median_mutual_information": float(np.median(mi)),

        "mean_mutual_information_correct": (
            float(mi[correct_mask].mean())
            if correct_mask.any()
            else float("nan")
        ),

        "mean_mutual_information_incorrect": (
            float(mi[error_mask].mean())
            if error_mask.any()
            else float("nan")
        ),

        "error_detection_auroc_mutual_information": (
            safe_error_detection_auroc(errors, mi)
        ),

        "error_detection_auroc_predictive_entropy": (
            safe_error_detection_auroc(errors, entropy)
        ),
    }


# ============================================================
# Main experiment
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("SensorGuard - MC-Dropout Uncertainty Experiments")
    print("=" * 78)
    print(f"Subset: {SUBSET}")
    print(f"Device: {DEVICE}")
    print(f"MC passes per window: {N_MC_PASSES}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    # 1. Load the original dataset.
    dataset = load_dataset(SUBSET)

    train_df = dataset["train"].copy()
    test_df = dataset["test"].copy()
    rul_df = dataset["rul"].copy()

    train_df = train_df.sort_values(
        ["engine_id", "cycle"],
        kind="stable",
    ).reset_index(drop=True)

    test_df = test_df.sort_values(
        ["engine_id", "cycle"],
        kind="stable",
    ).reset_index(drop=True)

    # 2. Reconstruct the original preprocessing pipeline.
    # The scaler is fitted on the training partition only.
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

    scaler = prepared["scaler"]

    # 3. Construct the official test labels.
    official_test = create_test_targets(
        test_df=test_df,
        rul_df=rul_df,
        horizon=HORIZON,
    )

    official_test = official_test.sort_values(
        ["engine_id", "cycle"],
        kind="stable",
    ).reset_index(drop=True)

    official_test_snapshot = official_test.copy(deep=True)

    # 4. Load the already-trained model.
    model = load_baseline(CHECKPOINT_PATH)

    # 5. Construct independent scenarios.
    scenarios = build_scenarios(official_test)

    if set(scenarios) != set(SCENARIOS):
        raise RuntimeError(
            "Generated scenario names do not match configuration."
        )

    summary_records = []
    prediction_frames = []

    reference_labels = None
    reference_engine_ids = None

    # 6. Run MC-Dropout on each scenario.
    for scenario_name in SCENARIOS:
        print(f"\nEvaluating: {scenario_name}")

        scenario_df = scenarios[scenario_name]

        if not scenario_df["engine_id"].equals(
            official_test["engine_id"]
        ):
            raise AssertionError(
                f"{scenario_name}: engine IDs changed."
            )

        if not scenario_df["cycle"].equals(
            official_test["cycle"]
        ):
            raise AssertionError(
                f"{scenario_name}: cycle values changed."
            )

        X, y, engine_ids, missing_count = (
            prepare_scenario_windows(
                scenario_df=scenario_df,
                scaler=scaler,
            )
        )

        uncertainty = mc_dropout_predict(
            model=model,
            X=X,
            n_passes=N_MC_PASSES,
            batch_size=BATCH_SIZE,
            device=DEVICE,
        )

        mean_probability = uncertainty["mean_probability"]

        if len(y) != len(mean_probability):
            raise AssertionError(
                f"{scenario_name}: prediction/label length mismatch."
            )

        if not np.isfinite(mean_probability).all():
            raise ValueError(
                f"{scenario_name}: non-finite mean probabilities."
            )

        # Verify that every scenario uses identical target ordering.
        if reference_labels is None:
            reference_labels = y.copy()
            reference_engine_ids = engine_ids.copy()
        else:
            if not np.array_equal(y, reference_labels):
                raise AssertionError(
                    f"{scenario_name}: target labels changed."
                )

            if not np.array_equal(
                engine_ids, reference_engine_ids
            ):
                raise AssertionError(
                    f"{scenario_name}: engine ordering changed."
                )

        summary = summarize_scenario(
            y_true=y,
            mean_probability=mean_probability,
            uncertainty=uncertainty,
        )

        summary_record = {
            "subset": SUBSET,
            "scenario": scenario_name,
            "engines": int(len(np.unique(engine_ids))),
            "missing_sensor_cells_before_imputation": missing_count,
            "mc_passes": N_MC_PASSES,
            **summary,
        }

        summary_records.append(summary_record)

        y_pred = (
            mean_probability >= THRESHOLD
        ).astype(int)

        # One row per input window.
        prediction_frames.append(
            pd.DataFrame(
                {
                    "subset": SUBSET,
                    "scenario": scenario_name,
                    "sample_index": np.arange(len(y)),
                    "engine_id": engine_ids.astype(int),
                    "y_true": y.astype(int),
                    "y_prob_mean": mean_probability,
                    "y_pred": y_pred,
                    "prediction_error": (
                        y_pred != y
                    ).astype(int),
                    "predictive_variance": (
                        uncertainty["predictive_variance"]
                    ),
                    "predictive_entropy": (
                        uncertainty["predictive_entropy"]
                    ),
                    "expected_entropy": (
                        uncertainty["expected_entropy"]
                    ),
                    "mutual_information": (
                        uncertainty["mutual_information"]
                    ),
                }
            )
        )

        print(
            f"  Accuracy={summary['accuracy']:.4f} | "
            f"F1={summary['f1']:.4f} | "
            f"AUROC={summary['auroc']:.4f} | "
            f"Brier={summary['brier']:.4f}"
        )

        print(
            f"  Mean MI={summary['mean_mutual_information']:.6f} | "
            f"Mean entropy={summary['mean_predictive_entropy']:.6f} | "
            f"Error rate={summary['error_rate']:.4f} | "
            f"Missing cells={missing_count:,}"
        )

    # 7. Confirm that original test data was not modified.
    pd.testing.assert_frame_equal(
        official_test,
        official_test_snapshot,
    )

    # 8. Save results.
    summary_df = pd.DataFrame(summary_records)

    summary_csv = (
        OUTPUT_DIR / "degradation_uncertainty_metrics.csv"
    )
    summary_json = (
        OUTPUT_DIR / "degradation_uncertainty_metrics.json"
    )
    predictions_csv = (
        OUTPUT_DIR / "degradation_uncertainty_predictions.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions_df.to_csv(
        predictions_csv,
        index=False,
    )

    report = {
        "project": "SensorGuard",
        "experiment": "mc_dropout_degradation_uncertainty",
        "subset": SUBSET,
        "checkpoint": str(CHECKPOINT_PATH),
        "device": str(DEVICE),
        "horizon": HORIZON,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "random_seed": RANDOM_SEED,
        "threshold": THRESHOLD,
        "mc_passes": N_MC_PASSES,
        "scaler_fitted_on_training_partition_only": True,
        "model_retrained": False,
        "official_test_modified": False,
        "missing_value_imputation": (
            "Missing sensor values are set to 0.0 after scaling."
        ),
        "uncertainty_definitions": {
            "predictive_variance": (
                "Sample variance of probabilities across MC passes."
            ),
            "predictive_entropy": (
                "Binary entropy of the mean predictive probability."
            ),
            "expected_entropy": (
                "Mean binary entropy across MC passes."
            ),
            "mutual_information": (
                "Predictive entropy minus expected entropy, "
                "clipped at zero for numerical stability."
            ),
            "error_detection_auroc": (
                "AUROC for ranking incorrect predictions by uncertainty; "
                "NaN when the error labels contain only one class."
            ),
        },
        "scenarios": summary_records,
    }

    with summary_json.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            allow_nan=True,
        )

    # 9. Display the most useful comparison columns.
    print("\n" + "=" * 100)
    print("MC-DROPOUT UNCERTAINTY RESULTS")
    print("=" * 100)

    display_columns = [
        "scenario",
        "accuracy",
        "f1",
        "brier",
        "mean_predictive_variance",
        "mean_predictive_entropy",
        "mean_mutual_information",
        "error_detection_auroc_mutual_information",
    ]

    print(
        summary_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    print("\nSaved files:")
    print(f"  Metrics CSV:     {summary_csv}")
    print(f"  Metrics JSON:    {summary_json}")
    print(f"  Predictions CSV: {predictions_csv}")

    print("\nMC-Dropout experiment completed successfully.")


if __name__ == "__main__":
    main()
