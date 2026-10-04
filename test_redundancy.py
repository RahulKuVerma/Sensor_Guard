from src.data.loader import load_dataset
from src.sensors.redundancy import (
    generate_redundancy_audit,
    print_redundancy_audit,
)


SUBSETS = [
    "FD001",
    "FD002",
    "FD003",
    "FD004",
]


def main():

    threshold = 0.90

    print("=" * 100)
    print("SensorGuard - Complete Sensor Redundancy Analysis")
    print("=" * 100)

    for subset in SUBSETS:

        print("\n" + "#" * 100)
        print(f"Processing {subset}")
        print("#" * 100)

        dataset = load_dataset(subset)

        train_df = dataset["train"]

        audit = generate_redundancy_audit(
            train_df,
            threshold=threshold,
        )

        print_redundancy_audit(
            audit,
            threshold=threshold,
        )

    print("\n" + "=" * 100)
    print("Complete redundancy analysis finished.")
    print("=" * 100)


if __name__ == "__main__":
    main()