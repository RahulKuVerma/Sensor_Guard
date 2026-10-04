from src.data.loader import load_dataset
from src.sensors.operating_conditions import (
    generate_operating_condition_audit,
    print_operating_condition_audit,
)


SUBSETS = [
    "FD001",
    "FD002",
    "FD003",
    "FD004",
]


def main():

    for subset in SUBSETS:

        print("\n")
        print("#" * 100)
        print(f"Processing {subset}")
        print("#" * 100)

        dataset = load_dataset(subset)

        train_df = dataset["train"]

        audit = (
            generate_operating_condition_audit(
                train_df
            )
        )

        print_operating_condition_audit(
            audit
        )


if __name__ == "__main__":
    main()