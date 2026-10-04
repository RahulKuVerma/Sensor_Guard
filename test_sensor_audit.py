from pathlib import Path

from src.data.loader import load_dataset
from src.sensors.analyzer import generate_sensor_audit


SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

RESULTS_DIR = Path("results") / "metrics"


def save_audit_results(
    subset: str,
    audit: dict,
):
    """
    Save all audit tables for one C-MAPSS subset.
    """

    subset_dir = RESULTS_DIR / subset
    subset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, dataframe in audit.items():

        output_file = (
            subset_dir
            / f"{name}.csv"
        )

        dataframe.to_csv(
            output_file,
            index=False,
        )

        print(
            f"Saved: {output_file}"
        )


def main():

    print("=" * 80)
    print("SensorGuard - Complete C-MAPSS Sensor Audit")
    print("=" * 80)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for subset in SUBSETS:

        print("\n" + "=" * 80)
        print(f"Processing {subset}")
        print("=" * 80)

        dataset = load_dataset(subset)

        train_df = dataset["train"]

        print(
            f"Training rows: {len(train_df)}"
        )

        print(
            f"Training engines: "
            f"{train_df['engine_id'].nunique()}"
        )

        audit = generate_sensor_audit(
            train_df
        )

        save_audit_results(
            subset,
            audit,
        )

    print("\n" + "=" * 80)
    print("Complete sensor audit finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()