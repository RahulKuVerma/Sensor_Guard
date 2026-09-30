from pathlib import Path

import pandas as pd


# ============================================================
# SensorGuard - C-MAPSS Dataset Loader
# ============================================================

# Project root:
# SensorGuard/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw C-MAPSS data directory
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# C-MAPSS column names
COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def get_file_path(subset: str, data_type: str) -> Path:
    """
    Return the path of a C-MAPSS file.

    Parameters
    ----------
    subset : str
        Dataset subset: FD001, FD002, FD003, or FD004.

    data_type : str
        One of:
        - train
        - test
        - rul

    Returns
    -------
    Path
        Full path to the requested file.
    """

    subset = subset.upper()
    data_type = data_type.lower()

    valid_subsets = {"FD001", "FD002", "FD003", "FD004"}
    valid_types = {"train", "test", "rul"}

    if subset not in valid_subsets:
        raise ValueError(
            f"Invalid subset '{subset}'. "
            f"Expected one of {sorted(valid_subsets)}."
        )

    if data_type not in valid_types:
        raise ValueError(
            f"Invalid data_type '{data_type}'. "
            f"Expected one of {sorted(valid_types)}."
        )

    file_name = f"{data_type}_{subset}.txt"

    return RAW_DATA_DIR / file_name


def load_train_data(subset: str) -> pd.DataFrame:
    """
    Load a C-MAPSS training dataset.

    Parameters
    ----------
    subset : str
        FD001, FD002, FD003, or FD004.

    Returns
    -------
    pandas.DataFrame
        Training data.
    """

    file_path = get_file_path(subset, "train")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Training file not found:\n{file_path}"
        )

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python",
    )

    return df


def load_test_data(subset: str) -> pd.DataFrame:
    """
    Load a C-MAPSS test dataset.

    Parameters
    ----------
    subset : str
        FD001, FD002, FD003, or FD004.

    Returns
    -------
    pandas.DataFrame
        Test data.
    """

    file_path = get_file_path(subset, "test")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Test file not found:\n{file_path}"
        )

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python",
    )

    return df


def load_rul_data(subset: str) -> pd.DataFrame:
    """
    Load the RUL file associated with a C-MAPSS test dataset.

    Parameters
    ----------
    subset : str
        FD001, FD002, FD003, or FD004.

    Returns
    -------
    pandas.DataFrame
        RUL values.
    """

    file_path = get_file_path(subset, "rul")

    if not file_path.exists():
        raise FileNotFoundError(
            f"RUL file not found:\n{file_path}"
        )

    rul = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["rul"],
        engine="python",
    )

    return rul


def load_dataset(subset: str) -> dict:
    """
    Load train, test, and RUL data for one C-MAPSS subset.

    Example
    -------
    dataset = load_dataset("FD001")

    train_df = dataset["train"]
    test_df = dataset["test"]
    rul_df = dataset["rul"]
    """

    subset = subset.upper()

    return {
        "train": load_train_data(subset),
        "test": load_test_data(subset),
        "rul": load_rul_data(subset),
    }


if __name__ == "__main__":
    # Simple standalone test
    subset = "FD001"

    print("=" * 60)
    print("SensorGuard - C-MAPSS Loader Test")
    print("=" * 60)

    dataset = load_dataset(subset)

    train_df = dataset["train"]
    test_df = dataset["test"]
    rul_df = dataset["rul"]

    print(f"\nSubset: {subset}")

    print("\nTrain shape:")
    print(train_df.shape)

    print("\nTest shape:")
    print(test_df.shape)

    print("\nRUL shape:")
    print(rul_df.shape)

    print("\nTrain columns:")
    print(train_df.columns.tolist())

    print("\nFirst 5 training rows:")
    print(train_df.head())

    print("\nFirst 5 RUL values:")
    print(rul_df.head())

    print("\nLoader test completed.")