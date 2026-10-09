
"""
SensorGuard - Validation-Fitted Temperature Scaling

Protocol:
1. Reconstruct the existing engine-level training/validation split.
2. Fit temperature scaling on clean validation MC-Dropout mean probabilities.
3. Evaluate the fixed temperature on official test data under six scenarios.
4. Save before/after metrics and per-window predictions.

No model retraining or test-set calibration fitting is performed.

Run:
    python -m src.evaluation.temperature_scaling_experiments

Methodological note:
Temperature scaling is applied to the logit of the MC-Dropout
mean probability. This calibrates the ensemble mean probability;
it is not equivalent to temperature-scaling each stochastic pass
before averaging.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import log_loss

from src.data.loader import load_dataset
from src.data.test_targets import create_test_targets
from src.features.dataset_builder import (
    prepare_training_validation_test,
)
from src.features.windows import create_sliding_windows
from src.models.baseline import create_baseline_model
from src.sensors.degradation import SensorDegradationEngine
from src.uncertainty.mc_dropout import mc_dropout_predict
from src.uncertainty.calibration import calibration_metrics


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

N_MC_PASSES = 50
BATCH_SIZE = 256
THRESHOLD = 0.50
N_BINS = 15

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
    / "temperature_scaling"
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
# Model loading
# ============================================================

def load_baseline(checkpoint_path: Path):
    """Load the existing checkpoint without retraining."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
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

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# Temperature scaling
# ============================================================

def fit_temperature(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """
    Fit a single positive temperature by minimizing validation
    binary log loss.

    Probabilities are converted to logits, then divided by T.
    """

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(
        probabilities, dtype=float
    ).reshape(-1)

    if len(y_true) == 0 or len(y_true) != len(probabilities):
        raise ValueError(
            "Validation labels and probabilities must be "
            "nonempty and have equal lengths."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Validation probabilities contain non-finite values."
        )

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Validation labels must be binary.")

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "Temperature fitting requires both validation classes."
        )

    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    raw_logits = logit(clipped)

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        scaled_probabilities = expit(
            raw_logits / temperature
        )
        scaled_probabilities = np.clip(
            scaled_probabilities, 1e-7, 1.0 - 1e-7
        )

        return float(
            log_loss(
                y_true,
                scaled_probabilities,
                labels=[0, 1],
            )
        )

    # Optimize log(T), which guarantees T > 0.
    result = minimize_scalar(
        objective,
        bounds=(-4.0, 4.0),
        method="bounded",
        options={"xatol": 1e-8},
    )

    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(
            f"Temperature optimization failed: {result.message}"
        )

    temperature = float(np.exp(result.x))

    return {
        "temperature": temperature,
        "validation_log_loss_before": objective(0.0),
        "validation_log_loss_after": float(result.fun),
        "optimizer_success": bool(result.success),
    }


def apply_temperature(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply the fixed temperature to probability logits."""

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")

    probabilities = np.asarray(
        probabilities, dtype=float
    )

    clipped = np.clip(
        probabilities, 1e-7, 1.0 - 1e-7
    )

    return expit(logit(clipped) / temperature)


# ============================================================
# Data preparation
# ============================================================

def prepare_test_windows(
    scenario_df: pd.DataFrame,
    scaler,
):
    """Scale corrupted raw test data and create model windows."""

    scaled = scaler.transform(scenario_df)

    values = scaled[SENSOR_COLUMNS].to_numpy(dtype=float)

    if np.isinf(values).any():
        raise ValueError("Infinite sensor values found after scaling.")

    missing_count = int(np.isnan(values).sum())

    scaled.loc[:, SENSOR_COLUMNS] = (
        scaled[SENSOR_COLUMNS].fillna(0.0)
    )

    X, y, engine_ids = create_sliding_windows(
        scaled,
        sensor_columns=SENSOR_COLUMNS,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    if len(X) == 0 or not np.isfinite(X).all():
        raise ValueError(
            "Invalid or empty test windows were generated."
        )

    return (
        X,
        np.asarray(y).astype(int),
        np.asarray(engine_ids),
        missing_count,
    )


def build_test_scenarios(
    official_test: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Create independent scenarios from the same test data."""

    engine = SensorDegradationEngine(
        random_seed=RANDOM_SEED
    )

    return {
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


# ============================================================
# Metrics and predictions
# ============================================================

def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """Evaluate discrimination, classification, and calibration."""

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if len(y_true) != len(probabilities):
        raise ValueError("Label and probability lengths differ.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite prediction probabilities.")

    probabilities = np.clip(
        probabilities, 1e-7, 1.0 - 1e-7
    )

    y_pred = (probabilities >= THRESHOLD).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    calibration = calibration_metrics(
        y_true,
        probabilities,
        n_bins=N_BINS,
    )

    return {
        "accuracy": float((y_pred == y_true).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier": calibration["brier"],
        "log_loss": calibration["log_loss"],
        "ece": calibration["ece"],
        "mce": calibration["mce"],
        "error_rate": float((y_pred != y_true).mean()),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "calibration_bins": calibration["calibration_bins"],
    }


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SensorGuard - Validation-Fitted Temperature Scaling")
    print("=" * 78)
    print(f"Subset: {SUBSET}")
    print(f"Device: {DEVICE}")
    print(f"MC passes: {N_MC_PASSES}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    # Load original dataset.
    dataset = load_dataset(SUBSET)

    train_df = dataset["train"].copy()
    test_df = dataset["test"].copy()
    rul_df = dataset["rul"].copy()

    train_df = train_df.sort_values(
        ["engine_id", "cycle"], kind="stable"
    ).reset_index(drop=True)

    test_df = test_df.sort_values(
        ["engine_id", "cycle"], kind="stable"
    ).reset_index(drop=True)

    # Reconstruct the existing engine-level split and scaler.
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

    # Validation data is already scaled by the existing pipeline.
    validation_df = prepared["validation_df"].copy()

    X_val, y_val, validation_engine_ids = create_sliding_windows(
        validation_df,
        sensor_columns=SENSOR_COLUMNS,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val).astype(int)

    if len(X_val) == 0 or not np.isfinite(X_val).all():
        raise ValueError("Validation windows are empty or non-finite.")

    if len(np.unique(y_val)) < 2:
        raise ValueError(
            "Validation windows must contain both classes."
        )

    print(f"\nValidation windows: {len(y_val):,}")
    print(
        "Validation engines:",
        len(np.unique(validation_engine_ids)),
    )

    model = load_baseline(CHECKPOINT_PATH)

    # Fit temperature only on clean validation predictions.
    print("\nEstimating validation MC-Dropout probabilities...")

    validation_uncertainty = mc_dropout_predict(
        model=model,
        X=X_val,
        n_passes=N_MC_PASSES,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    temperature_info = fit_temperature(
        y_true=y_val,
        probabilities=validation_uncertainty["mean_probability"],
    )

    temperature = temperature_info["temperature"]

    print(f"Fitted temperature: {temperature:.6f}")
    print(
        "Validation log loss: "
        f"{temperature_info['validation_log_loss_before']:.6f}"
        " -> "
        f"{temperature_info['validation_log_loss_after']:.6f}"
    )

    # Prepare official test labels and independent scenarios.
    official_test = create_test_targets(
        test_df=test_df,
        rul_df=rul_df,
        horizon=HORIZON,
    )

    official_test = official_test.sort_values(
        ["engine_id", "cycle"], kind="stable"
    ).reset_index(drop=True)

    original_snapshot = official_test.copy(deep=True)
    scenarios = build_test_scenarios(official_test)

    summary_records = []
    bin_records = []
    prediction_frames = []
    reference_labels = None
    reference_engine_ids = None

    for scenario in SCENARIOS:
        print(f"\nEvaluating test scenario: {scenario}")

        X, y, engine_ids, missing_count = prepare_test_windows(
            scenario_df=scenarios[scenario],
            scaler=scaler,
        )

        uncertainty = mc_dropout_predict(
            model=model,
            X=X,
            n_passes=N_MC_PASSES,
            batch_size=BATCH_SIZE,
            device=DEVICE,
        )

        raw_probabilities = uncertainty["mean_probability"]
        calibrated_probabilities = apply_temperature(
            raw_probabilities,
            temperature,
        )

        if reference_labels is None:
            reference_labels = y.copy()
            reference_engine_ids = engine_ids.copy()
        else:
            if not np.array_equal(y, reference_labels):
                raise AssertionError(
                    f"{scenario}: labels differ across scenarios."
                )
            if not np.array_equal(
                engine_ids, reference_engine_ids
            ):
                raise AssertionError(
                    f"{scenario}: engine ordering differs."
                )

        for method, probabilities in [
            ("uncalibrated", raw_probabilities),
            ("temperature_scaled", calibrated_probabilities),
        ]:
            metrics = evaluate_probabilities(y, probabilities)

            summary_records.append({
                "subset": SUBSET,
                "scenario": scenario,
                "method": method,
                "temperature": (
                    temperature if method == "temperature_scaled"
                    else 1.0
                ),
                "samples": len(y),
                "engines": len(np.unique(engine_ids)),
                "missing_sensor_cells_before_imputation": missing_count,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "calibration_bins"
                    and key != "confusion_matrix"
                },
                "confusion_matrix": metrics["confusion_matrix"],
            })

            for item in metrics["calibration_bins"]:
                bin_records.append({
                    "scenario": scenario,
                    "method": method,
                    **item,
                })

        y_pred_calibrated = (
            calibrated_probabilities >= THRESHOLD
        ).astype(int)

        prediction_frames.append(
            pd.DataFrame({
                "subset": SUBSET,
                "scenario": scenario,
                "sample_index": np.arange(len(y)),
                "engine_id": engine_ids.astype(int),
                "y_true": y,
                "probability_uncalibrated": raw_probabilities,
                "probability_temperature_scaled": calibrated_probabilities,
                "prediction_uncalibrated": (
                    raw_probabilities >= THRESHOLD
                ).astype(int),
                "prediction_temperature_scaled": y_pred_calibrated,
                "error_uncalibrated": (
                    (raw_probabilities >= THRESHOLD).astype(int) != y
                ).astype(int),
                "error_temperature_scaled": (
                    y_pred_calibrated != y
                ).astype(int),
                "mutual_information": uncertainty["mutual_information"],
                "predictive_entropy": uncertainty["predictive_entropy"],
            })
        )

        raw_metrics = evaluate_probabilities(y, raw_probabilities)
        cal_metrics = evaluate_probabilities(y, calibrated_probabilities)

        print(
            f"  ECE: {raw_metrics['ece']:.5f}"
            f" -> {cal_metrics['ece']:.5f} | "
            f"Log loss: {raw_metrics['log_loss']:.5f}"
            f" -> {cal_metrics['log_loss']:.5f}"
        )

    # Ensure no source test data was modified.
    pd.testing.assert_frame_equal(
        official_test,
        original_snapshot,
    )

    summary_df = pd.DataFrame(summary_records)
    bins_df = pd.DataFrame(bin_records)
    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    metrics_csv = OUTPUT_DIR / "temperature_scaling_metrics.csv"
    bins_csv = OUTPUT_DIR / "temperature_scaling_bins.csv"
    predictions_csv = OUTPUT_DIR / "temperature_scaling_predictions.csv"
    report_json = OUTPUT_DIR / "temperature_scaling_report.json"
    reliability_png = OUTPUT_DIR / "temperature_scaling_reliability.png"

    summary_df.to_csv(metrics_csv, index=False)
    bins_df.to_csv(bins_csv, index=False)
    predictions_df.to_csv(predictions_csv, index=False)

    report = {
        "project": "SensorGuard",
        "subset": SUBSET,
        "experiment": "validation_fitted_temperature_scaling",
        "checkpoint": str(CHECKPOINT_PATH),
        "device": str(DEVICE),
        "mc_passes": N_MC_PASSES,
        "validation_engine_split_seed": RANDOM_SEED,
        "temperature": temperature,
        "temperature_fit_data": "clean validation windows only",
        "test_used_for_temperature_fitting": False,
        "model_retrained": False,
        "temperature_method": (
            "Scalar temperature fitted to the logits of mean MC-Dropout "
            "probabilities using validation binary log loss."
        ),
        "validation_fit": temperature_info,
        "test_results": summary_records,
    }

    with report_json.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, allow_nan=False)

    # Reliability diagrams: separate panels for uncalibrated/calibrated
    # and a separate curve for each degradation scenario.
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 7), sharex=True, sharey=True
    )

    for ax, method in zip(
        axes,
        ["uncalibrated", "temperature_scaled"],
    ):
        ax.plot(
            [0, 1], [0, 1],
            linestyle="--",
            label="Perfect calibration",
        )

        for scenario in SCENARIOS:
            rows = bins_df[
                (bins_df["scenario"] == scenario)
                & (bins_df["method"] == method)
            ]

            if rows.empty:
                continue

            ax.plot(
                rows["mean_confidence"],
                rows["observed_frequency"],
                marker="o",
                markersize=3,
                label=scenario,
            )

        ax.set_title(method.replace("_", " ").title())
        ax.set_xlabel("Mean predicted probability")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Observed fault frequency")
    fig.suptitle(
        f"FD001 Reliability Diagrams: Before vs After Temperature Scaling"
    )
    fig.tight_layout()
    fig.savefig(reliability_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 78)
    print("TEMPERATURE SCALING RESULTS")
    print("=" * 78)

    print(
        summary_df[
            [
                "scenario", "method", "ece", "mce",
                "brier", "log_loss", "f1",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    print("\nSaved files:")
    print(f"  Metrics:       {metrics_csv}")
    print(f"  Calibration bins: {bins_csv}")
    print(f"  Predictions:   {predictions_csv}")
    print(f"  Report:        {report_json}")
    print(f"  Reliability:   {reliability_png}")

    print("\nTemperature-scaling evaluation completed.")


if __name__ == "__main__":
    main()
