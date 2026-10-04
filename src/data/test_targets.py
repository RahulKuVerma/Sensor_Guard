from typing import Tuple

import numpy as np
import pandas as pd


def add_test_rul(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct the true RUL for every observed cycle in the
    official C-MAPSS test set.

    For each engine:

        RUL(current cycle)
        =
        RUL at final observed cycle
        +
        (final observed cycle - current cycle)
    """

    required_test_columns = {
        "engine_id",
        "cycle",
    }

    missing = required_test_columns - set(test_df.columns)

    if missing:
        raise ValueError(
            f"Missing required test columns: {sorted(missing)}"
        )

    if "rul" not in rul_df.columns:
        raise ValueError(
            "RUL dataframe must contain a 'rul' column."
        )

    test = test_df.copy().sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)

    rul_values = rul_df["rul"].to_numpy(dtype=np.float64)

    engine_ids = sorted(test["engine_id"].unique())

    if len(engine_ids) != len(rul_values):
        raise ValueError(
            "Number of test engines does not match "
            "number of RUL values. "
            f"Engines={len(engine_ids)}, "
            f"RUL values={len(rul_values)}"
        )

    final_cycles = (
        test.groupby("engine_id")["cycle"]
        .max()
        .sort_index()
    )

    rul_mapping = dict(
        zip(engine_ids, rul_values)
    )

    test["final_observed_cycle"] = test["engine_id"].map(
        final_cycles
    )

    test["final_cycle_rul"] = test["engine_id"].map(
        rul_mapping
    )

    test["rul"] = (
        test["final_cycle_rul"]
        + (
            test["final_observed_cycle"]
            - test["cycle"]
        )
    )

    test["rul"] = test["rul"].astype(float)

    test.drop(
        columns=[
            "final_observed_cycle",
            "final_cycle_rul",
        ],
        inplace=True,
    )

    return test


def add_test_early_fault_target(
    test_df: pd.DataFrame,
    horizon: int = 30,
) -> pd.DataFrame:
    """
    Create the binary early-fault target for official test data.

    early_fault = 1 when RUL <= horizon.
    """

    if horizon < 0:
        raise ValueError(
            "Horizon must be non-negative."
        )

    if "rul" not in test_df.columns:
        raise ValueError(
            "Test dataframe must contain 'rul'. "
            "Run add_test_rul() first."
        )

    result = test_df.copy()

    result["early_fault"] = (
        result["rul"] <= horizon
    ).astype(np.int64)

    return result


def create_test_targets(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    horizon: int = 30,
) -> pd.DataFrame:
    """
    Complete official-test target construction.
    """

    result = add_test_rul(
        test_df=test_df,
        rul_df=rul_df,
    )

    result = add_test_early_fault_target(
        result,
        horizon=horizon,
    )

    return result


def validate_test_targets(
    test_df: pd.DataFrame,
) -> dict:
    """
    Validate reconstructed test RUL and early-fault labels.
    """

    required = {
        "engine_id",
        "cycle",
        "rul",
        "early_fault",
    }

    missing = required - set(test_df.columns)

    if missing:
        raise ValueError(
            f"Missing target columns: {sorted(missing)}"
        )

    rul_missing = int(
        test_df["rul"].isna().sum()
    )

    rul_negative = int(
        (test_df["rul"] < 0).sum()
    )

    target_values = sorted(
        test_df["early_fault"].unique().tolist()
    )

    target_valid = set(target_values).issubset({0, 1})

    return {
        "rows": len(test_df),
        "engines": test_df["engine_id"].nunique(),
        "missing_rul": rul_missing,
        "negative_rul": rul_negative,
        "target_values": target_values,
        "target_valid": target_valid,
        "valid": (
            rul_missing == 0
            and rul_negative == 0
            and target_valid
        ),
    }