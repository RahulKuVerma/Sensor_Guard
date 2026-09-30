from src.data.loader import load_dataset
from src.data.validator import (
    validate_dataframe,
    validate_rul,
    print_validation_report,
)


SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def main():

    print("=" * 70)
    print("SensorGuard - C-MAPSS Complete Dataset Validation")
    print("=" * 70)

    all_valid = True

    for subset in SUBSETS:

        print(f"\nLoading {subset}...")

        dataset = load_dataset(subset)

        train_df = dataset["train"]
        test_df = dataset["test"]
        rul_df = dataset["rul"]

        train_result = validate_dataframe(train_df)
        test_result = validate_dataframe(test_df)
        rul_result = validate_rul(rul_df)

        print_validation_report(
            subset,
            train_result,
            test_result,
            rul_result,
        )

        subset_valid = (
            train_result["valid"]
            and test_result["valid"]
            and rul_result["valid"]
        )

        if not subset_valid:
            all_valid = False

    print("\n" + "=" * 70)
    print("FINAL DATASET VALIDATION")
    print("=" * 70)

    if all_valid:
        print("All C-MAPSS subsets are VALID.")
    else:
        print("Some C-MAPSS subsets FAILED validation.")

    print("=" * 70)


if __name__ == "__main__":
    main()