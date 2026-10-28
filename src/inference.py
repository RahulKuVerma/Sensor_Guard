
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.loader import load_dataset
from src.data.preprocessing import create_training_dataset
from src.data.scaler import SensorScaler
from src.features.windows import split_engines
from src.models.baseline import create_baseline_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]

WINDOW_SIZE = 30
HORIZON = 30
THRESHOLD = 0.50

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.30

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def validate_upload(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an uploaded CSV and return a safe, sorted copy."""

    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    required = {"engine_id", "cycle", *SENSOR_COLUMNS}
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    result = df.copy()

    numeric_columns = ["engine_id", "cycle", *SENSOR_COLUMNS]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        )

    values = result[numeric_columns].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Required columns contain missing, non-numeric, "
            "or infinite values. Please correct the CSV."
        )

    if (result["cycle"] <= 0).any():
        raise ValueError("Cycle values must be positive.")

    if (result["engine_id"] <= 0).any():
        raise ValueError("Engine IDs must be positive.")

    if (result[SENSOR_COLUMNS].abs() > 1e12).any().any():
        raise ValueError(
            "Sensor values exceed the allowed numeric safety limit."
        )

    if result.duplicated(["engine_id", "cycle"]).any():
        raise ValueError(
            "Duplicate engine_id/cycle rows were found. "
            "Each engine must have at most one row per cycle."
        )

    result["engine_id"] = result["engine_id"].astype(int)
    result["cycle"] = result["cycle"].astype(int)

    return result.sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)


def load_matching_scaler(subset: str) -> SensorScaler:
    """Reconstruct the scaler using the original training split."""

    if subset not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError(f"Unsupported C-MAPSS subset: {subset}")

    dataset = load_dataset(subset)
    train_df = dataset["train"]

    training = create_training_dataset(
        train_df,
        horizon=HORIZON,
    )

    train_split, _ = split_engines(
        training,
        train_ratio=0.85,
        validation_ratio=0.15,
        random_seed=42,
    )

    scaler = SensorScaler(sensor_columns=SENSOR_COLUMNS)
    scaler.fit_transform(train_split)

    return scaler


def load_saved_model(subset: str) -> torch.nn.Module:
    """Load an existing checkpoint without training."""

    checkpoint_path = (
        PROJECT_ROOT
        / "results"
        / "models"
        / subset
        / "baseline_lstm.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Saved model checkpoint not found: {checkpoint_path}"
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


def predict_uploaded_engines(
    uploaded_df: pd.DataFrame,
    subset: str,
) -> tuple[pd.DataFrame, int]:
    """
    Predict the RUL-derived early-fault target for each eligible engine.

    Uses the latest 30 cycles per engine. Engines with fewer than
    30 cycles are skipped. Returns the prediction table and skipped
    engine count.
    """

    df = validate_upload(uploaded_df)

    scaler = load_matching_scaler(subset)
    model = load_saved_model(subset)

    scaled_df = scaler.transform(df)

    windows = []
    engine_ids = []
    latest_cycles = []
    skipped_engines = 0

    for engine_id, engine_df in scaled_df.groupby(
        "engine_id", sort=True
    ):
        engine_df = engine_df.sort_values("cycle")

        if len(engine_df) < WINDOW_SIZE:
            skipped_engines += 1
            continue

        latest = engine_df.tail(WINDOW_SIZE)

        window = latest[SENSOR_COLUMNS].to_numpy(
            dtype=np.float32
        )

        if window.shape != (WINDOW_SIZE, len(SENSOR_COLUMNS)):
            raise ValueError(
                f"Unexpected window shape for engine {engine_id}."
            )

        if not np.isfinite(window).all():
            raise ValueError(
                f"Invalid scaled sensor values for engine {engine_id}."
            )

        windows.append(window)
        engine_ids.append(int(engine_id))
        latest_cycles.append(int(latest["cycle"].iloc[-1]))

    if not windows:
        raise ValueError(
            "No engine has at least 30 cycles. "
            "Upload more readings for at least one engine."
        )

    X = torch.as_tensor(
        np.stack(windows),
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.inference_mode():
        logits = model(X)
        probabilities = torch.sigmoid(logits).cpu().numpy()

    results = pd.DataFrame({
        "engine_id": engine_ids,
        "latest_cycle": latest_cycles,
        "early_fault_probability": probabilities,
        "predicted_class": (
            probabilities >= THRESHOLD
        ).astype(int),
    })

    results["predicted_label"] = results["predicted_class"].map({
        0: "Not flagged",
        1: "Early-fault target flagged",
    })

    return results, skipped_engines
