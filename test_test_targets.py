from src.data.loader import load_dataset
from src.data.test_targets import (
    create_test_targets,
    validate_test_targets,
)


SUBSETS = [
    "FD001",
    "FD002",
    "FD003",
    "FD004",
]


def main():
    print("=" * 100)
    print("SensorGuard - Official C-MAPSS Test Target Validation")
    print("=" * 100)

    for subset in SUBSETS:
        print("\n" + "#" * 100)
        print(f"Processing {subset}")
        print("#" * 100)

        dataset = load_dataset(subset)

        test_df = create_test_targets(
            dataset["test"],
            dataset["rul"],
            horizon=30,
        )

        report = validate_test_targets(
            test_df
        )

        print(f"Rows: {report['rows']}")
        print(f"Engines: {report['engines']}")
        print(f"Missing RUL: {report['missing_rul']}")
        print(f"Negative RUL: {report['negative_rul']}")
        print(
            f"Target values: "
            f"{report['target_values']}"
        )
        print(
            f"Target valid: "
            f"{report['target_valid']}"
        )
        print(
            f"Validation: "
            f"{report['valid']}"
        )

        print("\nRUL statistics:")
        print(
            test_df["rul"]
            .describe()
        )

        print("\nEarly-fault distribution:")
        print(
            test_df["early_fault"]
            .value_counts()
            .sort_index()
        )

        assert report["valid"]

    print("\n" + "=" * 100)
    print("Official test target construction completed successfully.")
    print("=" * 100)


if __name__ == "__main__":
    main()