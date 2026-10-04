from src.data.loader import load_dataset
from src.data.preprocessing import (
    create_training_dataset,
)
from src.features.windows import (
    create_sliding_windows,
    split_engines,
    summarize_windows,
    verify_engine_split,
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


def main():

    WINDOW_SIZE = 30
    STRIDE = 1
    HORIZON = 30

    print("=" * 100)
    print(
        "SensorGuard - Leakage-Controlled "
        "Sliding Window Generation"
    )
    print("=" * 100)

    for subset in SUBSETS:

        print("\n" + "#" * 100)
        print(f"Processing {subset}")
        print("#" * 100)

        dataset = load_dataset(
            subset
        )

        train_df = (
            create_training_dataset(
                dataset["train"],
                horizon=HORIZON,
            )
        )

        (
            train_df,
            validation_df,
            test_df,
        ) = split_engines(
            train_df,
            train_ratio=0.70,
            validation_ratio=0.15,
            random_seed=42,
        )

        split_result = (
            verify_engine_split(
                train_df,
                validation_df,
                test_df,
            )
        )

        print("\nEngine Split")
        print("-" * 100)

        print(
            f"Train engines: "
            f"{split_result['train_engines']}"
        )

        print(
            f"Validation engines: "
            f"{split_result['validation_engines']}"
        )

        print(
            f"Test engines: "
            f"{split_result['test_engines']}"
        )

        print(
            f"Leakage check: "
            f"{split_result['valid']}"
        )

        if not split_result["valid"]:
            raise RuntimeError(
                "Engine leakage detected!"
            )

        # ----------------------------------------------------
        # Create windows
        # ----------------------------------------------------

        X_train, y_train, train_engine_ids = (
            create_sliding_windows(
                train_df,
                SENSOR_COLUMNS,
                window_size=WINDOW_SIZE,
                stride=STRIDE,
            )
        )

        (
            X_validation,
            y_validation,
            validation_engine_ids,
        ) = create_sliding_windows(
            validation_df,
            SENSOR_COLUMNS,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
        )

        X_test, y_test, test_engine_ids = (
            create_sliding_windows(
                test_df,
                SENSOR_COLUMNS,
                window_size=WINDOW_SIZE,
                stride=STRIDE,
            )
        )

        # ----------------------------------------------------
        # Summaries
        # ----------------------------------------------------

        train_summary = summarize_windows(
            X_train,
            y_train,
            train_engine_ids,
        )

        validation_summary = summarize_windows(
            X_validation,
            y_validation,
            validation_engine_ids,
        )

        test_summary = summarize_windows(
            X_test,
            y_test,
            test_engine_ids,
        )

        print("\nWindow Shapes")
        print("-" * 100)

        print(
            f"Train:      {X_train.shape}"
        )

        print(
            f"Validation: {X_validation.shape}"
        )

        print(
            f"Test:       {X_test.shape}"
        )

        print("\nWindow Summary")
        print("-" * 100)

        print(
            f"Train windows: "
            f"{train_summary['samples']}"
        )

        print(
            f"Validation windows: "
            f"{validation_summary['samples']}"
        )

        print(
            f"Test windows: "
            f"{test_summary['samples']}"
        )

        print(
            f"Train early-fault ratio: "
            f"{train_summary['early_fault_ratio']:.4f}"
        )

        print(
            f"Validation early-fault ratio: "
            f"{validation_summary['early_fault_ratio']:.4f}"
        )

        print(
            f"Test early-fault ratio: "
            f"{test_summary['early_fault_ratio']:.4f}"
        )

        print(
            f"\n{HORIZON}-cycle early-fault target: OK"
        )

        print(
            f"Window size: "
            f"{WINDOW_SIZE}"
        )

        print(
            f"Stride: "
            f"{STRIDE}"
        )

    print("\n" + "=" * 100)
    print(
        "Leakage-controlled window generation "
        "completed successfully."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()