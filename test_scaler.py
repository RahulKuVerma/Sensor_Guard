from src.data.loader import load_dataset
from src.data.preprocessing import create_training_dataset
from src.data.scaler import SensorScaler


SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
]


def main():
    print("=" * 100)
    print("SensorGuard - Leakage-Safe Sensor Scaling Test")
    print("=" * 100)

    dataset = load_dataset("FD001")

    train_df = create_training_dataset(
        dataset["train"],
        horizon=30,
    )

    scaler = SensorScaler(SENSOR_COLUMNS)

    scaled_train = scaler.fit_transform(
        train_df
    )

    print("\nOriginal sensor means:")
    print(
        train_df[SENSOR_COLUMNS]
        .mean()
        .head()
    )

    print("\nScaled sensor means:")
    print(
        scaled_train[SENSOR_COLUMNS]
        .mean()
        .head()
    )

    print("\nScaled sensor standard deviations:")
    print(
        scaled_train[SENSOR_COLUMNS]
        .std()
        .head()
    )

    # Verify approximate zero mean.
    max_abs_mean = (
        scaled_train[SENSOR_COLUMNS]
        .mean()
        .abs()
        .max()
    )

    print(
        f"\nMaximum absolute scaled mean: "
        f"{max_abs_mean:.8f}"
    )

    assert max_abs_mean < 1e-6

    print("\nScaler test: PASS")
    print("=" * 100)


if __name__ == "__main__":
    main()