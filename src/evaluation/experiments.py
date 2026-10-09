
"""
SensorGuard - Model-Based Degradation Experiments

Phase 16, Step 2:
Evaluate the existing FD001 baseline under controlled
synthetic sensor degradation.

Important:
- Does not retrain the model.
- Does not modify raw C-MAPSS files.
- Fits the scaler on the original training partition only.
- Applies degradation to a copy of the official test data.
- Imputes missing scaled sensor values with 0.0, corresponding
  to the training mean under StandardScaler.
- Uses the same test labels for every scenario.
- Does not tune the model or decision threshold on test data.

Run from the project root:
    python -m src.evaluation.degradation_experiments
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import load_dataset
from src.data.test_targets import create_test_targets
from src.features.dataset_builder import (
    prepare_training_validation_test,
)
from src.features.windows import create_sliding_windows
from src.models.baseline import create_baseline_model
from src.models.dataset import create_dataloader
from src.evaluation.metrics import compute_classification_metrics
from src.sensors.degradation import SensorDegradationEngine


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
    / "degradation_experiments"
)

SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
]

SCENARIOS = [
    "clean",
    "random_dropout",
    "missing_interval",
    "gaussian_noise",
    "positive_bias",
    "positive_drift",
]


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load Existing Baseline
# ============================================================

def load_baseline(checkpoint_path: Path):
    """Load the trained baseline without retraining."""

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

    # This is a trusted, locally generated project checkpoint.
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with PyTorch versions that do not
        # support the weights_only argument.
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
        # Also support checkpoints containing only model weights.
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# Build Degradation Scenarios
# ============================================================

def build_scenarios(
    official_test: pd.DataFrame,
) -> dict:
    """
    Create independent degraded copies of the same test dataframe.

    Every scenario starts from the original, non-degraded data.
    """

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
# Prepare Model Inputs
# ============================================================

def prepare_scenario_windows(
    scenario_df: pd.DataFrame,
    scaler,
):
    """
    Scale a scenario with the training-fitted scaler and create
    windows using the project's existing window builder.

    Missing observations are imputed with zero AFTER scaling.
    Under StandardScaler, zero corresponds to the training mean.
    """

    scaled = scaler.transform(scenario_df)

    sensor_values = scaled[SENSOR_COLUMNS].to_numpy(
        dtype=np.float64
    )

    missing_mask = np.isnan(sensor_values)

    if np.isinf(sensor_values).any():
        raise ValueError(
            "Infinite sensor values found after scaling."
        )

    # Explicit missing-data handling for the model.
    # Zero is the standardized training-mean value.
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
            "No windows were produced. Check trajectory lengths "
            "and window configuration."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Model windows contain NaN or infinite values."
        )

    return X, y, engine_ids, int(missing_mask.sum())


# ============================================================
# Model Predictions
# ============================================================

@torch.no_grad()
def predict_probabilities(
    model,
    X: np.ndarray,
) -> np.ndarray:
    """Generate baseline probabilities without model training."""

    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    probabilities = []

    model.eval()

    for (X_batch,) in loader:
        X_batch = X_batch.to(DEVICE)

        logits = model(X_batch)
        probs = torch.sigmoid(logits)

        probabilities.append(
            probs.detach().cpu().numpy()
        )

    return np.concatenate(probabilities).astype(float)


# ============================================================
# Main Experiment
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("SensorGuard - Model-Based Degradation Experiments")
    print("=" * 78)
    print(f"Subset: {SUBSET}")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    # --------------------------------------------------------
    # 1. Load original C-MAPSS data
    # --------------------------------------------------------

    dataset = load_dataset(SUBSET)

    train_df = dataset["train"].copy()
    test_df = dataset["test"].copy()
    rul_df = dataset["rul"].copy()

    # Keep trajectories chronological and reset index labels.
    train_df = (
        train_df.sort_values(
            ["engine_id", "cycle"],
            kind="stable",
        ).reset_index(drop=True)
    )

    test_df = (
        test_df.sort_values(
            ["engine_id", "cycle"],
            kind="stable",
        ).reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 2. Reconstruct the baseline preprocessing pipeline
    # --------------------------------------------------------

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

    # This scaler is fitted only on the original training split.
    scaler = prepared["scaler"]

    # Construct official test labels and RUL targets.
    official_test = create_test_targets(
        test_df=test_df,
        rul_df=rul_df,
        horizon=HORIZON,
    )

    official_test = (
        official_test.sort_values(
            ["engine_id", "cycle"],
            kind="stable",
        ).reset_index(drop=True)
    )

    # Snapshot raw inputs to check that scenarios do not mutate them.
    official_test_snapshot = official_test.copy(deep=True)

    # --------------------------------------------------------
    # 3. Load the existing baseline
    # --------------------------------------------------------

    model = load_baseline(CHECKPOINT_PATH)

    # --------------------------------------------------------
    # 4. Generate scenarios
    # --------------------------------------------------------

    scenarios = build_scenarios(official_test)

    if set(scenarios) != set(SCENARIOS):
        raise RuntimeError(
            "Generated scenario names do not match configuration."
        )

    results = []
    predictions_by_scenario = {}

    print("\nRunning scenarios...\n")

    # --------------------------------------------------------
    # 5. Evaluate each scenario
    # --------------------------------------------------------

    reference_labels = None
    reference_engine_ids = None

    for scenario_name in SCENARIOS:
        print(f"Evaluating: {scenario_name}")

        scenario_df = scenarios[scenario_name]

        # Check metadata preservation.
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

        probabilities = predict_probabilities(
            model=model,
            X=X,
        )

        y = np.asarray(y).astype(int)
        engine_ids = np.asarray(engine_ids)

        if len(y) != len(probabilities):
            raise AssertionError(
                f"{scenario_name}: prediction/label length mismatch."
            )

        # All scenarios must use exactly the same label ordering.
        if reference_labels is None:
            reference_labels = y.copy()
            reference_engine_ids = engine_ids.copy()
        else:
            if not np.array_equal(y, reference_labels):
                raise AssertionError(
                    f"{scenario_name}: labels differ from clean scenario."
                )

            if not np.array_equal(
                engine_ids,
                reference_engine_ids,
            ):
                raise AssertionError(
                    f"{scenario_name}: engine ordering differs."
                )

        metrics = compute_classification_metrics(
            y_true=y,
            y_prob=probabilities,
            threshold=THRESHOLD,
        )

        predictions_by_scenario[scenario_name] = {
            "y_true": y,
            "y_prob": probabilities,
            "engine_ids": engine_ids,
        }

        results.append(
            {
                "subset": SUBSET,
                "scenario": scenario_name,
                "samples": int(len(y)),
                "engines": int(len(np.unique(engine_ids))),
                "missing_sensor_cells_before_imputation": (
                    missing_count
                ),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "brier": metrics["brier"],
                "threshold": metrics["threshold"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
        )

        print(
            f"  Accuracy={metrics['accuracy']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"AUROC={metrics['auroc']:.4f} | "
            f"AUPRC={metrics['auprc']:.4f} | "
            f"Brier={metrics['brier']:.4f} | "
            f"Missing cells={missing_count:,}"
        )

    # --------------------------------------------------------
    # 6. Verify the original data was not changed
    # --------------------------------------------------------

    pd.testing.assert_frame_equal(
        official_test,
        official_test_snapshot,
    )

    # --------------------------------------------------------
    # 7. Save metrics
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    csv_path = OUTPUT_DIR / "degradation_metrics.csv"
    json_path = OUTPUT_DIR / "degradation_metrics.json"
    predictions_path = OUTPUT_DIR / "degradation_predictions.csv"

    results_df.to_csv(
        csv_path,
        index=False,
    )

    report = {
        "project": "SensorGuard",
        "experiment": "model_based_degradation",
        "subset": SUBSET,
        "checkpoint": str(CHECKPOINT_PATH),
        "device": str(DEVICE),
        "horizon": HORIZON,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "random_seed": RANDOM_SEED,
        "threshold": THRESHOLD,
        "scaler_fitted_on_training_partition_only": True,
        "model_retrained": False,
        "official_test_modified": False,
        "missing_value_imputation": (
            "After scaling, missing sensor values are set to 0.0, "
            "equivalent to the training mean under StandardScaler."
        ),
        "scenarios": results,
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    prediction_records = []

    for scenario_name, values in predictions_by_scenario.items():
        for index in range(len(values["y_true"])):
            prediction_records.append(
                {
                    "subset": SUBSET,
                    "scenario": scenario_name,
                    "sample_index": index,
                    "engine_id": int(
                        values["engine_ids"][index]
                    ),
                    "y_true": int(
                        values["y_true"][index]
                    ),
                    "y_prob": float(
                        values["y_prob"][index]
                    ),
                    "y_pred": int(
                        values["y_prob"][index] >= THRESHOLD
                    ),
                }
            )

    pd.DataFrame(
        prediction_records
    ).to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # 8. Print comparison
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("DEGRADATION EXPERIMENT RESULTS")
    print("=" * 78)

    display_columns = [
        "scenario",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "brier",
    ]

    print(
        results_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved files:")
    print(f"  Metrics CSV:     {csv_path}")
    print(f"  Metrics JSON:    {json_path}")
    print(f"  Predictions CSV: {predictions_path}")

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
