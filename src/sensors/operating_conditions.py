import numpy as np
import pandas as pd


# ============================================================
# SensorGuard - Operating Condition Analysis
# ============================================================

SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
]

SETTING_COLUMNS = [
    "setting_1",
    "setting_2",
    "setting_3",
]


def analyze_setting_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate basic statistics for the three operating settings.
    """

    records = []

    for setting in SETTING_COLUMNS:

        if setting not in df.columns:
            continue

        series = df[setting]

        records.append(
            {
                "setting": setting,
                "mean": series.mean(),
                "std": series.std(),
                "variance": series.var(),
                "min": series.min(),
                "max": series.max(),
                "unique_values": series.nunique(),
            }
        )

    return pd.DataFrame(records)


def analyze_sensor_setting_correlation(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate absolute Pearson correlation between every sensor
    and every operating setting.
    """

    records = []

    for sensor in SENSOR_COLUMNS:

        if sensor not in df.columns:
            continue

        for setting in SETTING_COLUMNS:

            if setting not in df.columns:
                continue

            sensor_std = df[sensor].std()
            setting_std = df[setting].std()

            if sensor_std == 0 or setting_std == 0:
                correlation = np.nan
            else:
                correlation = df[sensor].corr(
                    df[setting]
                )

            records.append(
                {
                    "sensor": sensor,
                    "setting": setting,
                    "correlation": correlation,
                    "absolute_correlation": (
                        abs(correlation)
                        if pd.notna(correlation)
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(records)


def calculate_max_setting_sensitivity(
    correlation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find the strongest operating-setting relationship
    for every sensor.
    """

    if correlation_df.empty:
        return pd.DataFrame(
            columns=[
                "sensor",
                "max_setting_correlation",
                "most_sensitive_setting",
            ]
        )

    records = []

    for sensor, group in correlation_df.groupby("sensor"):

        valid_group = group.dropna(
            subset=["absolute_correlation"]
        )

        if valid_group.empty:
            records.append(
                {
                    "sensor": sensor,
                    "max_setting_correlation": np.nan,
                    "most_sensitive_setting": None,
                }
            )
            continue

        strongest = valid_group.loc[
            valid_group["absolute_correlation"].idxmax()
        ]

        records.append(
            {
                "sensor": sensor,
                "max_setting_correlation": (
                    strongest["absolute_correlation"]
                ),
                "most_sensitive_setting": (
                    strongest["setting"]
                ),
            }
        )

    return pd.DataFrame(records)


def generate_operating_condition_audit(
    df: pd.DataFrame,
) -> dict:
    """
    Generate the complete operating-condition audit.
    """

    setting_statistics = analyze_setting_statistics(
        df
    )

    sensor_setting_correlation = (
        analyze_sensor_setting_correlation(df)
    )

    sensor_sensitivity = (
        calculate_max_setting_sensitivity(
            sensor_setting_correlation
        )
    )

    return {
        "setting_statistics": setting_statistics,
        "sensor_setting_correlation": (
            sensor_setting_correlation
        ),
        "sensor_setting_sensitivity": (
            sensor_sensitivity
        ),
    }


def print_operating_condition_audit(
    audit: dict,
):
    """
    Print operating-condition analysis.
    """

    print("\n" + "=" * 100)
    print(
        "SensorGuard - Operating Condition Audit"
    )
    print("=" * 100)

    print("\nOperating Setting Statistics")
    print("-" * 100)

    print(
        audit[
            "setting_statistics"
        ].to_string(index=False)
    )

    print(
        "\nSensor / Operating Setting Correlation"
    )
    print("-" * 100)

    correlation_table = (
        audit[
            "sensor_setting_correlation"
        ]
        .sort_values(
            "absolute_correlation",
            ascending=False,
            na_position="last",
        )
    )

    print(
        correlation_table.to_string(
            index=False
        )
    )

    print(
        "\nMaximum Operating-Condition Sensitivity"
    )
    print("-" * 100)

    sensitivity_table = (
        audit[
            "sensor_setting_sensitivity"
        ]
        .sort_values(
            "max_setting_correlation",
            ascending=False,
            na_position="last",
        )
    )

    print(
        sensitivity_table.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print(
        "Operating-condition audit completed."
    )
    print("=" * 100)