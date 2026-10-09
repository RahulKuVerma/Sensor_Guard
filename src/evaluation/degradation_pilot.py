
"""
SensorGuard - Controlled Sensor Degradation Pilot

Phase 16, Step 1:
Apply synthetic degradation mechanisms to copies of the
C-MAPSS training data and save an audit summary.

Run from the SensorGuard project root:
    python -m src.evaluation.degradation_pilot
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_dataset
from src.sensors.degradation import SensorDegradationEngine


# ============================================================
# Configuration
# ============================================================

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
]

RANDOM_SEED = 42

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "metrics"
    / "degradation_pilot"
)


# ============================================================
# Degradation Scenarios
# ============================================================

def apply_scenarios(
    train_df: pd.DataFrame,
) -> dict:
    """
    Generate independent degraded copies of training data.

    The original dataframe is never modified.

    Each scenario starts from the same clean dataframe.
    """

    engine = SensorDegradationEngine(
        random_seed=RANDOM_SEED
    )

    sensors = SENSOR_COLUMNS.copy()

    return {
        "clean": train_df.copy(),

        "random_dropout": engine.random_dropout(
            train_df,
            sensors=sensors,
            dropout_rate=0.10,
        ),

        "missing_interval": engine.missing_interval(
            train_df,
            sensors=sensors,
            interval_length=10,
            num_intervals=1,
        ),

        "gaussian_noise": engine.add_noise(
            train_df,
            sensors=sensors,
            noise_std=0.20,
        ),

        "positive_bias": engine.add_bias(
            train_df,
            sensors=sensors,
            bias_scale=0.20,
            direction="positive",
        ),

        "positive_drift": engine.add_drift(
            train_df,
            sensors=sensors,
            drift_scale=0.50,
            direction="positive",
        ),
    }


# ============================================================
# Scenario Audit
# ============================================================

def audit_scenario(
    original: pd.DataFrame,
    degraded: pd.DataFrame,
    subset: str,
    scenario: str,
) -> dict:
    """
    Measure the effect of a scenario on sensor observations.

    Reports missing values, changed cells, affected rows,
    engine count, and basic integrity checks.
    """

    original_sensors = original[SENSOR_COLUMNS]
    degraded_sensors = degraded[SENSOR_COLUMNS]

    # A cell is changed if its value differs, including
    # finite-to-NaN and NaN-to-finite transitions.
    original_values = original_sensors.to_numpy(dtype=float)
    degraded_values = degraded_sensors.to_numpy(dtype=float)

    equal_cells = (
        np.isclose(
            original_values,
            degraded_values,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )
    )

    changed_cells = int((~equal_cells).sum())

    missing_cells = int(
        degraded_sensors.isna().to_numpy().sum()
    )

    affected_rows = int(
        (~equal_cells).any(axis=1).sum()
    )

    # Check that metadata and dataframe shape are preserved.
    metadata_columns = [
        column
        for column in ("engine_id", "cycle")
        if column in original.columns
    ]

    metadata_unchanged = all(
        original[column].equals(degraded[column])
        for column in metadata_columns
    )

    shape_unchanged = original.shape == degraded.shape

    # Non-missing sensor values should remain finite.
    sensor_values_finite = bool(
        np.isfinite(
            degraded_sensors.to_numpy(dtype=float)
        ).all()
        or np.isfinite(
            degraded_sensors.to_numpy(dtype=float)
        )[~degraded_sensors.isna().to_numpy()].all()
    )

    return {
        "subset": subset,
        "scenario": scenario,
        "rows": int(len(degraded)),
        "engines": int(degraded["engine_id"].nunique()),
        "sensor_channels": len(SENSOR_COLUMNS),
        "changed_sensor_cells": changed_cells,
        "missing_sensor_cells": missing_cells,
        "affected_rows": affected_rows,
        "changed_cell_fraction": (
            changed_cells / original_values.size
            if original_values.size
            else 0.0
        ),
        "metadata_unchanged": bool(metadata_unchanged),
        "shape_unchanged": bool(shape_unchanged),
        "sensor_values_valid": sensor_values_finite,
    }


# ============================================================
# Main Pilot
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_records = []

    print("=" * 78)
    print("SensorGuard - Controlled Sensor Degradation Pilot")
    print("=" * 78)

    for subset in SUBSETS:
        print(f"\nProcessing {subset}...")

        # Load the original C-MAPSS data.
        dataset = load_dataset(subset)
        original = dataset["train"].copy()

        # Validate required columns.
        required_columns = [
            "engine_id",
            "cycle",
            *SENSOR_COLUMNS,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in original.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{subset} is missing required columns: "
                f"{missing_columns}"
            )

        if original["engine_id"].isna().any():
            raise ValueError(
                f"{subset} contains missing engine IDs."
            )

        # Ensure each engine's rows follow chronological order.
        original = (
            original
            .sort_values(
                ["engine_id", "cycle"],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        original_snapshot = original.copy(deep=True)

        scenarios = apply_scenarios(original)

        for scenario_name, degraded in scenarios.items():
            record = audit_scenario(
                original=original,
                degraded=degraded,
                subset=subset,
                scenario=scenario_name,
            )

            # Every scenario must preserve the data structure.
            if not record["shape_unchanged"]:
                raise AssertionError(
                    f"{subset}/{scenario_name}: "
                    "dataframe shape changed."
                )

            if not record["metadata_unchanged"]:
                raise AssertionError(
                    f"{subset}/{scenario_name}: "
                    "engine IDs or cycles changed."
                )

            if not record["sensor_values_valid"]:
                raise AssertionError(
                    f"{subset}/{scenario_name}: "
                    "invalid sensor values detected."
                )

            all_records.append(record)

            print(
                f"  {scenario_name:20s} "
                f"changed={record['changed_sensor_cells']:>9,} "
                f"missing={record['missing_sensor_cells']:>8,}"
            )

        # Confirm that applying scenarios did not mutate
        # the clean training dataframe.
        pd.testing.assert_frame_equal(
            original,
            original_snapshot,
        )

    # ========================================================
    # Save Results
    # ========================================================

    summary_df = pd.DataFrame(all_records)

    csv_path = OUTPUT_DIR / "degradation_pilot_summary.csv"
    json_path = OUTPUT_DIR / "degradation_pilot_summary.json"

    summary_df.to_csv(
        csv_path,
        index=False,
    )

    report = {
        "project": "SensorGuard",
        "experiment": "controlled_degradation_pilot",
        "random_seed": RANDOM_SEED,
        "subsets": SUBSETS,
        "scenarios": [
            "clean",
            "random_dropout",
            "missing_interval",
            "gaussian_noise",
            "positive_bias",
            "positive_drift",
        ],
        "data_source": "NASA C-MAPSS training data",
        "official_test_modified": False,
        "original_data_modified": False,
        "records": all_records,
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

    print("\n" + "=" * 78)
    print("DEGRADATION PILOT COMPLETED")
    print("=" * 78)

    print(f"Subsets processed: {len(SUBSETS)}")
    print(f"Scenarios per subset: 6")
    print(f"Total audit records: {len(all_records)}")
    print(f"\nCSV summary:  {csv_path}")
    print(f"JSON report:  {json_path}")

    print("\nIntegrity checks:")
    print(
        "  Shape preserved:       ",
        bool(summary_df["shape_unchanged"].all()),
    )
    print(
        "  Metadata preserved:    ",
        bool(summary_df["metadata_unchanged"].all()),
    )
    print(
        "  Sensor values valid:   ",
        bool(summary_df["sensor_values_valid"].all()),
    )

    print("\nNext: inspect the summary before model evaluation.")


if __name__ == "__main__":
    main()
