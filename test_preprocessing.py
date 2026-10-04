from src.data.loader import load_dataset
from src.data.preprocessing import (
    create_training_dataset,
    summarize_target,
    validate_rul_target,
)


SUBSETS = [
    "FD001",
    "FD002",
    "FD003",
    "FD004",
]


def main():

    horizon = 30

    print("=" * 100)
    print(
        "SensorGuard - Early Fault Target Construction"
    )
    print("=" * 100)

    for subset in SUBSETS:

        print("\n" + "#" * 100)
        print(f"Processing {subset}")
        print("#" * 100)

        dataset = load_dataset(subset)

        train_df = dataset["train"]

        processed = create_training_dataset(
            train_df,
            horizon=horizon,
        )

        validation = validate_rul_target(
            processed
        )

        summary = summarize_target(
            processed
        )

        print("\nTarget Validation")
        print("-" * 100)

        print(
            f"Valid: {validation['valid']}"
        )

        print(
            f"Rows: {validation['rows']}"
        )

        print(
            f"Missing RUL: "
            f"{validation['missing_rul']}"
        )

        print(
            f"Negative RUL: "
            f"{validation['negative_rul']}"
        )

        print(
            f"Target values: "
            f"{validation['target_values']}"
        )

        print("\nTarget Distribution")
        print("-" * 100)

        print(
            f"Normal rows: "
            f"{summary['normal_rows']}"
        )

        print(
            f"Early-fault rows: "
            f"{summary['early_fault_rows']}"
        )

        print(
            f"Early-fault ratio: "
            f"{summary['early_fault_ratio']:.4f}"
        )

    print("\n" + "=" * 100)
    print(
        "Early-fault target construction completed."
    )
    print("=" * 100)


if __name__ == "__main__":
    main()