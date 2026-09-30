from src.data.loader import load_dataset


SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def main():
    print("=" * 70)
    print("SensorGuard - C-MAPSS All Subsets Loader Test")
    print("=" * 70)

    for subset in SUBSETS:
        print(f"\nTesting {subset}...")

        dataset = load_dataset(subset)

        train_df = dataset["train"]
        test_df = dataset["test"]
        rul_df = dataset["rul"]

        print(f"  Train   : {train_df.shape}")
        print(f"  Test    : {test_df.shape}")
        print(f"  RUL     : {rul_df.shape}")
        print(f"  Columns : {len(train_df.columns)}")

    print("\n" + "=" * 70)
    print("All C-MAPSS subsets loaded successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()