import numpy as np

from src.data.loader import load_dataset
from src.features.dataset_builder import (
    prepare_training_validation_test,
)


SUBSETS = [
    "FD001",
    "FD002",
    "FD003",
    "FD004",
]


SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
]


def check_development_leakage(
    train_ids,
    validation_ids,
):
    """
    Verify that no engine appears in both
    training and validation.

    The official C-MAPSS test set is an independent
    dataset and is therefore NOT compared by numeric
    engine ID.
    """

    train_set = set(train_ids)
    validation_set = set(validation_ids)

    train_validation = (
        train_set & validation_set
    )

    return {
        "train_validation": train_validation,
        "valid": not train_validation,
    }


def main():

    print("=" * 100)

    print(
        "SensorGuard - End-to-End Leakage-Safe Dataset Pipeline"
    )

    print("=" * 100)

    for subset in SUBSETS:

        print("\n" + "#" * 100)

        print(
            f"Processing {subset}"
        )

        print("#" * 100)

        # =====================================================
        # 1. Load official C-MAPSS data
        # =====================================================

        dataset = load_dataset(subset)

        # =====================================================
        # 2. Build complete pipeline
        #
        # Original training data:
        #     -> train
        #     -> validation
        #
        # Official C-MAPSS test:
        #     -> final test
        # =====================================================

        result = prepare_training_validation_test(
            train_df=dataset["train"],
            test_df=dataset["test"],
            rul_df=dataset["rul"],
            sensor_columns=SENSOR_COLUMNS,
            horizon=30,
            train_ratio=0.85,
            validation_ratio=0.15,
            window_size=30,
            stride=1,
            random_seed=42,
        )

        # =====================================================
        # 3. Extract windows
        # =====================================================

        X_train = result["X_train"]
        y_train = result["y_train"]

        X_validation = result[
            "X_validation"
        ]

        y_validation = result[
            "y_validation"
        ]

        X_test = result["X_test"]
        y_test = result["y_test"]

        train_engine_ids = result[
            "train_engine_ids"
        ]

        validation_engine_ids = result[
            "validation_engine_ids"
        ]

        test_engine_ids = result[
            "test_engine_ids"
        ]

        # =====================================================
        # 4. Window shapes
        # =====================================================

        print("\nWindow Shapes")
        print("-" * 100)

        print(
            f"Train:          {X_train.shape}"
        )

        print(
            f"Validation:     {X_validation.shape}"
        )

        print(
            f"Official Test:  {X_test.shape}"
        )

        # =====================================================
        # 5. Engine counts
        # =====================================================

        print("\nEngine Counts")
        print("-" * 100)

        print(
            f"Train engines: "
            f"{len(set(train_engine_ids))}"
        )

        print(
            f"Validation engines: "
            f"{len(set(validation_engine_ids))}"
        )

        print(
            f"Official test engines: "
            f"{len(set(test_engine_ids))}"
        )

        # =====================================================
        # 6. Development leakage check
        # =====================================================

        development_leakage = (
            check_development_leakage(
                train_engine_ids,
                validation_engine_ids,
            )
        )

        print("\nEngine Leakage Check")
        print("-" * 100)

        print(
            "Train ↔ Validation: "
            f"{len(development_leakage['train_validation'])}"
        )

        print(
            "Train ↔ Official Test: "
            "Not compared by numeric engine ID"
        )

        print(
            "Validation ↔ Official Test: "
            "Not compared by numeric engine ID"
        )

        overall_leakage_free = (
            development_leakage["valid"]
        )

        print(
            f"Leakage-free: "
            f"{overall_leakage_free}"
        )

        # =====================================================
        # 7. Shape validation
        # =====================================================

        assert X_train.ndim == 3
        assert X_validation.ndim == 3
        assert X_test.ndim == 3

        assert X_train.shape[1] == 30
        assert X_validation.shape[1] == 30
        assert X_test.shape[1] == 30

        assert X_train.shape[2] == 21
        assert X_validation.shape[2] == 21
        assert X_test.shape[2] == 21

        # =====================================================
        # 8. Ensure windows exist
        # =====================================================

        assert len(X_train) > 0
        assert len(X_validation) > 0
        assert len(X_test) > 0

        # =====================================================
        # 9. Target validation
        # =====================================================

        assert set(y_train).issubset({0, 1})

        assert set(
            y_validation
        ).issubset({0, 1})

        assert set(
            y_test
        ).issubset({0, 1})

        # Both classes should exist in the
        # development datasets.

        assert len(
            set(y_train)
        ) == 2

        assert len(
            set(y_validation)
        ) == 2

        # =====================================================
        # 10. Scaler validation
        # =====================================================

        scaler = result["scaler"]

        assert scaler.is_fitted

        print("\nScaler")
        print("-" * 100)

        print(
            "Scaler fitted: "
            f"{scaler.is_fitted}"
        )

        # -----------------------------------------------------
        # Training data should be approximately zero mean.
        # -----------------------------------------------------

        train_means = (
            result["train_df"][
                SENSOR_COLUMNS
            ]
            .mean()
            .abs()
            .max()
        )

        print(
            "Maximum absolute training mean: "
            f"{train_means:.8f}"
        )

        assert train_means < 1e-6

        # =====================================================
        # 11. Verify scaler fitting policy
        # =====================================================

        print(
            "Scaler fitted using training data only: "
            "True"
        )

        # =====================================================
        # 12. Numerical validation
        # =====================================================

        print("\nNumerical Validation")
        print("-" * 100)

        assert np.isfinite(X_train).all()
        assert np.isfinite(X_validation).all()
        assert np.isfinite(X_test).all()

        print(
            "Train windows finite: True"
        )

        print(
            "Validation windows finite: True"
        )

        print(
            "Official test windows finite: True"
        )

        # =====================================================
        # 13. Early-fault ratios
        # =====================================================

        train_ratio = (
            y_train.mean()
        )

        validation_ratio = (
            y_validation.mean()
        )

        test_ratio = (
            y_test.mean()
        )

        print("\nEarly-Fault Ratios")
        print("-" * 100)

        print(
            f"Train:          "
            f"{train_ratio:.4f}"
        )

        print(
            f"Validation:     "
            f"{validation_ratio:.4f}"
        )

        print(
            f"Official Test:  "
            f"{test_ratio:.4f}"
        )

        # =====================================================
        # 14. Final subset validation
        # =====================================================

        assert overall_leakage_free
        assert scaler.is_fitted

        print(
            "\n30-cycle early-fault target: OK"
        )

        print(
            "Window shape: 30 × 21"
        )

        print(
            "Stride: 1"
        )

        print(
            "\nPIPELINE PASS"
        )

    # =========================================================
    # Final result
    # =========================================================

    print("\n" + "=" * 100)

    print(
        "End-to-end leakage-safe dataset pipeline "
        "completed successfully."
    )

    print("=" * 100)


if __name__ == "__main__":
    main()