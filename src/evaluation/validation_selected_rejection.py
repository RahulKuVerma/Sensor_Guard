
"""
SensorGuard: Validation-selected uncertainty rejection.

Protocol
--------
1. Reconstruct the existing train/validation/test preprocessing split.
2. Run MC-Dropout on CLEAN validation windows only.
3. Select the uncertainty measure with the lowest validation error
   among the least-uncertain engines at a target 80% coverage.
4. Freeze the selected measure and its numeric validation threshold.
5. Apply that unchanged policy to all six existing official-test scenarios.
6. Compare with random engine selection at matched test coverage.
7. Bootstrap whole test engines while keeping the policy fixed.

This script does not retrain the model, alter checkpoints, or tune on test.

Run:
    python -m src.evaluation.validation_selected_rejection
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.loader import load_dataset
from src.features.dataset_builder import prepare_training_validation_test
from src.models.baseline import create_baseline_model
from src.uncertainty.mc_dropout import mc_dropout_predict


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

SCENARIOS = [
    "clean",
    "random_dropout",
    "missing_interval",
    "gaussian_noise",
    "positive_bias",
    "positive_drift",
]

UNCERTAINTY_COLUMNS = {
    "mutual_information": "mutual_information",
    "predictive_variance": "predictive_variance",
    "predictive_entropy": "predictive_entropy",
}

HORIZON = 30
WINDOW_SIZE = 30
STRIDE = 1

TRAIN_RATIO = 0.85
VALIDATION_RATIO = 0.15
SEED = 42

MC_PASSES = 50
BATCH_SIZE = 256
CLASSIFICATION_THRESHOLD = 0.50

# Choose the rejection policy using validation engines only.
TARGET_VALIDATION_COVERAGE = 0.80

# Resampling whole engines, not individual windows.
BOOTSTRAP_REPEATS = 500
RANDOM_REPEATS = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]

OUTPUT_DIR = ROOT / "results" / "metrics" / "validation_selected_rejection"


# ------------------------------------------------------------
# Model and preparation
# ------------------------------------------------------------

def load_model(subset: str):
    checkpoint_path = (
        ROOT / "results" / "models" / subset / "baseline_lstm.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for {subset}: {checkpoint_path}"
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
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    return model


def first_available(prepared, names, subset, description):
    for name in names:
        if name in prepared:
            return np.asarray(prepared[name])

    available = ", ".join(sorted(prepared.keys()))
    raise KeyError(
        f"{subset}: could not find {description}. "
        f"Tried {names}. Available keys: {available}"
    )


def prepare_subset(subset: str):
    dataset = load_dataset(subset)

    train_df = dataset["train"].copy()
    test_df = dataset["test"].copy()
    rul_df = dataset["rul"].copy()

    train_df = train_df.sort_values(
        ["engine_id", "cycle"], kind="stable"
    ).reset_index(drop=True)

    test_df = test_df.sort_values(
        ["engine_id", "cycle"], kind="stable"
    ).reset_index(drop=True)

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
        random_seed=SEED,
    )

    X_val = first_available(
        prepared,
        ["X_val", "X_validation", "X_valid", "X_validation_windows"],
        subset,
        "validation features",
    )
    y_val = first_available(
        prepared,
        ["y_val", "y_validation", "y_valid", "y_validation_windows"],
        subset,
        "validation labels",
    ).astype(int).reshape(-1)

    val_ids = first_available(
        prepared,
        [
            "val_engine_ids",
            "validation_engine_ids",
            "valid_engine_ids",
            "engine_ids_val",
            "validation_ids",
        ],
        subset,
        "validation engine IDs",
    ).reshape(-1)

    if not (len(X_val) == len(y_val) == len(val_ids)):
        raise ValueError(
            f"{subset}: validation features, labels, and engine IDs "
            "have inconsistent lengths."
        )

    if not np.isfinite(X_val).all():
        raise ValueError(f"{subset}: validation windows contain invalid values.")

    return X_val, y_val, val_ids


# ------------------------------------------------------------
# Engine-level aggregation
# ------------------------------------------------------------

def aggregate_engine_records(
    y_true,
    y_pred,
    engine_ids,
    uncertainty_scores,
):
    """
    Aggregate per-window data to one record per engine.

    Engine uncertainty is the mean uncertainty across its windows.
    Engine error rate is the fraction of its windows misclassified.
    """
    frame = pd.DataFrame({
        "engine_id": np.asarray(engine_ids).astype(str),
        "y_true": np.asarray(y_true).astype(int),
        "y_pred": np.asarray(y_pred).astype(int),
    })

    frame["error"] = (frame["y_true"] != frame["y_pred"]).astype(int)

    for name, scores in uncertainty_scores.items():
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if len(scores) != len(frame):
            raise ValueError(f"{name}: uncertainty length mismatch.")
        if not np.isfinite(scores).all():
            raise ValueError(f"{name}: non-finite uncertainty values.")
        frame[name] = scores

    aggregations = {
        "n_windows": ("error", "size"),
        "error_rate": ("error", "mean"),
        "errors": ("error", "sum"),
        "fault_windows": ("y_true", "sum"),
        "predicted_fault_windows": ("y_pred", "sum"),
    }

    for name in uncertainty_scores:
        aggregations[f"score_{name}"] = (name, "mean")

    return (
        frame.groupby("engine_id", sort=False)
        .agg(**aggregations)
        .reset_index()
    )


def risk_metrics(engines):
    """Calculate retained engine and pooled-window risk."""
    if len(engines) == 0:
        return {
            "retained_engines": 0,
            "retained_windows": 0,
            "engine_coverage": np.nan,
            "pooled_window_error_rate": np.nan,
            "mean_engine_error_rate": np.nan,
            "fault_recall": np.nan,
            "predicted_fault_rate": np.nan,
        }

    windows = int(engines["n_windows"].sum())
    errors = int(engines["errors"].sum())
    fault_windows = int(engines["fault_windows"].sum())
    predicted_fault_windows = int(
        engines["predicted_fault_windows"].sum()
    )

    return {
        "retained_engines": int(len(engines)),
        "retained_windows": windows,
        "pooled_window_error_rate": errors / windows if windows else np.nan,
        "mean_engine_error_rate": float(engines["error_rate"].mean()),
        "fault_recall": (
            float(
                engines["fault_windows"].sum()
                - 0
            ) if False else (
                float(
                    engines["fault_windows"].sum()
                )
            )
        ),
        "predicted_fault_rate": (
            predicted_fault_windows / windows if windows else np.nan
        ),
    }


def finalize_risk_metrics(engines, total_engines):
    result = risk_metrics(engines)
    result["engine_coverage"] = (
        len(engines) / total_engines if total_engines else np.nan
    )

    true_faults = int(engines["fault_windows"].sum()) if len(engines) else 0
    # fault_windows counts actual positives, so recall requires TP.
    # Derive TP from each engine's labels and predictions in the caller
    # and pass the correct value through the extra column when available.
    if "true_positives" in engines.columns:
        total_positive = int(engines["fault_windows"].sum())
        result["fault_recall"] = (
            int(engines["true_positives"].sum()) / total_positive
            if total_positive else np.nan
        )
    else:
        result["fault_recall"] = np.nan

    return result


def add_true_positive_counts(frame, y_true, y_pred, engine_ids):
    detail = pd.DataFrame({
        "engine_id": np.asarray(engine_ids).astype(str),
        "tp": (
            (np.asarray(y_true) == 1)
            & (np.asarray(y_pred) == 1)
        ).astype(int),
    })
    counts = (
        detail.groupby("engine_id", sort=False)["tp"]
        .sum()
        .rename("true_positives")
        .reset_index()
    )
    return frame.merge(counts, on="engine_id", how="left")


# ------------------------------------------------------------
# Policy selection on validation only
# ------------------------------------------------------------

def select_validation_policy(validation_engines):
    """
    At the pre-specified 80% engine coverage, choose the metric with
    the lowest pooled window error on validation data.

    Threshold is the numeric uncertainty score at the last retained
    validation engine. The same threshold is applied unchanged to test.
    """
    n = len(validation_engines)
    if n < 2:
        raise ValueError("Too few validation engines for rejection selection.")

    k = max(1, int(np.ceil(TARGET_VALIDATION_COVERAGE * n)))
    candidates = []

    for metric in UNCERTAINTY_COLUMNS:
        score_col = f"score_{metric}"
        ranked = validation_engines.sort_values(
            score_col, ascending=True, kind="mergesort"
        )
        retained = ranked.iloc[:k].copy()
        score_threshold = float(retained[score_col].max())
        metrics = finalize_risk_metrics(retained, n)

        candidates.append({
            "metric": metric,
            "score_column": score_col,
            "threshold": score_threshold,
            "nominal_coverage": TARGET_VALIDATION_COVERAGE,
            "validation_actual_coverage": len(retained) / n,
            "validation_retained_window_error": (
                metrics["pooled_window_error_rate"]
            ),
            "validation_mean_engine_error": (
                metrics["mean_engine_error_rate"]
            ),
            "validation_retained_engines": len(retained),
            "validation_total_engines": n,
        })

    # Deterministic tie-breaking: lowest error, then higher coverage,
    # then fixed metric order.
    candidates.sort(
        key=lambda x: (
            x["validation_retained_window_error"],
            -x["validation_actual_coverage"],
            list(UNCERTAINTY_COLUMNS).index(x["metric"]),
        )
    )

    return candidates[0], pd.DataFrame(candidates)


# ------------------------------------------------------------
# Test evaluation with frozen threshold
# ------------------------------------------------------------

def bootstrap_frozen_policy(
    engines,
    score_column,
    threshold,
    repeats,
    rng,
):
    """
    Resample complete test engines and apply the fixed validation threshold.
    The policy is never refitted or reranked inside the test bootstrap.
    """
    n = len(engines)
    risks = []

    for _ in range(repeats):
        idx = rng.integers(0, n, size=n)
        sample = engines.iloc[idx].reset_index(drop=True)
        retained = sample[sample[score_column] <= threshold]
        if len(retained) == 0:
            continue
        metrics = finalize_risk_metrics(retained, len(sample))
        risks.append(metrics["pooled_window_error_rate"])

    if not risks:
        return np.nan, np.nan

    lo, hi = np.quantile(risks, [0.025, 0.975])
    return float(lo), float(hi)


def random_engine_baseline(engines, n_retained, repeats, rng):
    """Randomly retain the same number of engines as the frozen policy."""
    n = len(engines)
    k = max(1, min(n, int(n_retained)))

    if k == n:
        m = finalize_risk_metrics(engines, n)
        return m["pooled_window_error_rate"], m["mean_engine_error_rate"]

    window_risks = []
    engine_risks = []

    for _ in range(repeats):
        idx = rng.choice(n, size=k, replace=False)
        selected = engines.iloc[idx]
        metrics = finalize_risk_metrics(selected, n)
        window_risks.append(metrics["pooled_window_error_rate"])
        engine_risks.append(metrics["mean_engine_error_rate"])

    return float(np.mean(window_risks)), float(np.mean(engine_risks))


def load_test_scenario(path, subset, scenario):
    frame = pd.read_csv(path)
    frame = frame[frame["scenario"] == scenario].copy()

    if frame.empty:
        raise ValueError(f"{subset}/{scenario}: no prediction rows found.")

    # Existing experiment output uses these column names.
    required = {
        "engine_id",
        "y_true",
        "y_pred",
        *UNCERTAINTY_COLUMNS.values(),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{subset}/{scenario}: missing required columns {missing}"
        )

    return frame


def evaluate_test_scenario(frame, subset, scenario, policy, rng):
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_pred = frame["y_pred"].to_numpy(dtype=int)
    engine_ids = frame["engine_id"].to_numpy()

    scores = {
        metric: frame[column].to_numpy(dtype=float)
        for metric, column in UNCERTAINTY_COLUMNS.items()
    }

    engines = aggregate_engine_records(
        y_true, y_pred, engine_ids, scores
    )
    engines = add_true_positive_counts(
        engines, y_true, y_pred, engine_ids
    )

    score_column = policy["score_column"]
    threshold = float(policy["threshold"])

    # Frozen policy: no threshold or metric selection on test.
    retained = engines[engines[score_column] <= threshold].copy()

    if retained.empty:
        retained_metrics = finalize_risk_metrics(retained, len(engines))
        boot_low, boot_high = np.nan, np.nan
        random_window, random_engine = np.nan, np.nan
    else:
        retained_metrics = finalize_risk_metrics(retained, len(engines))
        boot_low, boot_high = bootstrap_frozen_policy(
            engines,
            score_column,
            threshold,
            BOOTSTRAP_REPEATS,
            rng,
        )
        random_window, random_engine = random_engine_baseline(
            engines,
            len(retained),
            RANDOM_REPEATS,
            rng,
        )

    full_metrics = finalize_risk_metrics(engines, len(engines))

    return {
        "subset": subset,
        "scenario": scenario,
        "selected_uncertainty": policy["metric"],
        "frozen_validation_threshold": threshold,
        "test_engines": len(engines),
        "test_windows": int(engines["n_windows"].sum()),
        "retained_engines": len(retained),
        "actual_test_engine_coverage": (
            len(retained) / len(engines) if len(engines) else np.nan
        ),
        "validation_selected_nominal_coverage": TARGET_VALIDATION_COVERAGE,
        "full_test_window_error_rate": (
            full_metrics["pooled_window_error_rate"]
        ),
        "retained_window_error_rate": (
            retained_metrics["pooled_window_error_rate"]
        ),
        "retained_mean_engine_error_rate": (
            retained_metrics["mean_engine_error_rate"]
        ),
        "retained_fault_recall": retained_metrics["fault_recall"],
        "retained_predicted_fault_rate": (
            retained_metrics["predicted_fault_rate"]
        ),
        "random_matched_window_error_rate": random_window,
        "random_matched_mean_engine_error_rate": random_engine,
        "error_reduction_vs_random": (
            random_window - retained_metrics["pooled_window_error_rate"]
            if np.isfinite(random_window)
            and np.isfinite(retained_metrics["pooled_window_error_rate"])
            else np.nan
        ),
        "bootstrap_window_error_ci_low": boot_low,
        "bootstrap_window_error_ci_high": boot_high,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    policy_records = []
    candidate_records = []
    test_records = []
    audit_records = []

    print("=" * 78)
    print("SensorGuard — Validation-Selected Uncertainty Rejection")
    print("=" * 78)
    print(f"Device: {DEVICE}")
    print(f"MC passes: {MC_PASSES}")
    print(f"Target validation engine coverage: {TARGET_VALIDATION_COVERAGE:.0%}")
    print("Test policy will remain frozen.")

    for subset in SUBSETS:
        print(f"\n--- {subset} ---")

        try:
            X_val, y_val, val_ids = prepare_subset(subset)
            model = load_model(subset)

            val_uncertainty = mc_dropout_predict(
                model=model,
                X=X_val,
                n_passes=MC_PASSES,
                batch_size=BATCH_SIZE,
                device=DEVICE,
            )

            val_prob = val_uncertainty["mean_probability"]
            val_pred = (
                val_prob >= CLASSIFICATION_THRESHOLD
            ).astype(int)

            validation_scores = {
                metric: val_uncertainty[column]
                for metric, column in UNCERTAINTY_COLUMNS.items()
            }

            val_engines = aggregate_engine_records(
                y_val, val_pred, val_ids, validation_scores
            )
            val_engines = add_true_positive_counts(
                val_engines, y_val, val_pred, val_ids
            )

            policy, candidates = select_validation_policy(val_engines)
            policy["subset"] = subset
            policy["validation_engine_count"] = len(val_engines)
            policy["validation_window_count"] = len(y_val)

            policy_records.append(policy)

            candidates.insert(0, "subset", subset)
            candidate_records.append(candidates)

            print(
                f"Selected on validation: {policy['metric']} | "
                f"threshold={policy['threshold']:.8g} | "
                f"validation error="
                f"{policy['validation_retained_window_error']:.4f}"
            )

            prediction_path = (
                ROOT / "results" / "metrics" / subset
                / "mc_dropout_experiments"
                / "degradation_uncertainty_predictions.csv"
            )

            if not prediction_path.exists():
                raise FileNotFoundError(
                    f"Existing MC-Dropout predictions not found: "
                    f"{prediction_path}"
                )

            for scenario in SCENARIOS:
                test_frame = load_test_scenario(
                    prediction_path, subset, scenario
                )

                result = evaluate_test_scenario(
                    test_frame, subset, scenario, policy, rng
                )
                test_records.append(result)

                print(
                    f"  {scenario:18s} "
                    f"coverage={result['actual_test_engine_coverage']:.3f} "
                    f"error={result['retained_window_error_rate']:.4f} "
                    f"random={result['random_matched_window_error_rate']:.4f} "
                    f"fault_recall={result['retained_fault_recall']:.4f}"
                )

            audit_records.append({
                "subset": subset,
                "status": "success",
                "validation_windows": len(y_val),
                "validation_engines": len(val_engines),
                "test_scenarios_evaluated": len(SCENARIOS),
                "selected_metric": policy["metric"],
                "frozen_threshold": policy["threshold"],
            })

        except Exception as exc:
            print(f"[ERROR] {subset}: {exc}")
            audit_records.append({
                "subset": subset,
                "status": f"failed: {exc}",
            })

    if not test_records:
        raise RuntimeError(
            "No test results were generated. Review the input audit."
        )

    policies_df = pd.DataFrame(policy_records)
    candidates_df = (
        pd.concat(candidate_records, ignore_index=True)
        if candidate_records else pd.DataFrame()
    )
    results_df = pd.DataFrame(test_records)
    audit_df = pd.DataFrame(audit_records)

    policies_df.to_csv(
        OUTPUT_DIR / "selected_validation_policies.csv", index=False
    )
    candidates_df.to_csv(
        OUTPUT_DIR / "validation_policy_candidates.csv", index=False
    )
    results_df.to_csv(
        OUTPUT_DIR / "frozen_policy_test_results.csv", index=False
    )
    audit_df.to_csv(
        OUTPUT_DIR / "input_audit.csv", index=False
    )

    report = {
        "project": "SensorGuard",
        "experiment": "validation_selected_uncertainty_rejection",
        "policy_selection_data": "clean validation engines only",
        "test_threshold_tuning": False,
        "model_retrained": False,
        "target_validation_engine_coverage": TARGET_VALIDATION_COVERAGE,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "mc_passes": MC_PASSES,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "random_baseline_repeats": RANDOM_REPEATS,
        "policies": policy_records,
        "test_results": test_records,
        "audit": audit_records,
    }

    with (OUTPUT_DIR / "report.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2, allow_nan=True)

    display_cols = [
        "subset",
        "scenario",
        "selected_uncertainty",
        "actual_test_engine_coverage",
        "retained_window_error_rate",
        "random_matched_window_error_rate",
        "retained_fault_recall",
        "error_reduction_vs_random",
        "bootstrap_window_error_ci_low",
        "bootstrap_window_error_ci_high",
    ]

    print("\n" + "=" * 110)
    print("FROZEN-POLICY TEST RESULTS")
    print("=" * 110)
    print(
        results_df[display_cols].round(4).to_string(index=False)
    )

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
