from src.data.loader import load_dataset
from src.sensors.analyzer import (
    generate_sensor_audit,
    print_sensor_audit,
)


def main():

    subset = "FD001"

    print(f"Loading {subset}...")

    dataset = load_dataset(subset)

    train_df = dataset["train"]

    audit = generate_sensor_audit(train_df)

    print_sensor_audit(audit)


if __name__ == "__main__":
    main()