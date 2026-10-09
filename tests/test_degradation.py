
import numpy as np
import pandas as pd

from src.sensors.degradation import SensorDegradationEngine


# ============================================================
# Dummy Dataset
# ============================================================

def create_dummy_data():
    """Create a C-MAPSS-like dataframe with two 50-cycle engines."""
    return pd.DataFrame(
        {
            "engine_id": np.repeat([1, 2], 50),
            "cycle": list(range(1, 51)) * 2,
            **{
                f"sensor_{i}": np.arange(
                    100, 200, dtype=float
                )
                for i in range(1, 22)
            },
        }
    )


# ============================================================
# Random Dropout Tests
# ============================================================

def test_random_dropout():
    df = create_dummy_data()
    original = df.copy()

    degraded = SensorDegradationEngine(
        random_seed=42
    ).random_dropout(
        df,
        sensors=["sensor_1"],
        dropout_rate=0.20,
    )

    pd.testing.assert_frame_equal(df, original)
    pd.testing.assert_series_equal(
        df["sensor_2"], degraded["sensor_2"]
    )

    missing_count = int(
        degraded["sensor_1"].isna().sum()
    )

    assert 0 < missing_count < len(df)
    assert 0.10 <= missing_count / len(df) <= 0.30

    print("Random dropout test: PASS")


def test_reproducibility():
    df = create_dummy_data()

    result_1 = SensorDegradationEngine(
        random_seed=42
    ).random_dropout(
        df, sensors=["sensor_1"], dropout_rate=0.20
    )

    result_2 = SensorDegradationEngine(
        random_seed=42
    ).random_dropout(
        df, sensors=["sensor_1"], dropout_rate=0.20
    )

    pd.testing.assert_frame_equal(result_1, result_2)
    print("Reproducibility test: PASS")


def test_invalid_rate():
    df = create_dummy_data()
    engine = SensorDegradationEngine()

    for invalid_rate in (-0.1, 1.1, np.nan, np.inf):
        try:
            engine.random_dropout(
                df,
                sensors=["sensor_1"],
                dropout_rate=invalid_rate,
            )
        except ValueError:
            continue

        raise AssertionError(
            f"Invalid dropout rate {invalid_rate!r} was accepted."
        )

    print("Invalid rate test: PASS")


def test_invalid_sensor():
    df = create_dummy_data()

    try:
        SensorDegradationEngine().random_dropout(
            df,
            sensors=["sensor_999"],
            dropout_rate=0.20,
        )
    except ValueError:
        print("Invalid sensor test: PASS")
        return

    raise AssertionError("Invalid sensor was accepted.")


# ============================================================
# Missing Interval Tests
# ============================================================

def test_missing_interval():
    df = create_dummy_data()
    original = df.copy()

    degraded = SensorDegradationEngine(
        random_seed=42
    ).missing_interval(
        df,
        sensors=["sensor_1"],
        interval_length=10,
        num_intervals=2,
    )

    pd.testing.assert_frame_equal(df, original)
    pd.testing.assert_series_equal(
        df["sensor_2"], degraded["sensor_2"]
    )

    assert int(degraded["sensor_1"].isna().sum()) == 40

    for _, group in degraded.groupby(
        "engine_id", sort=False
    ):
        missing = group["sensor_1"].isna().to_numpy()
        assert int(missing.sum()) == 20

        longest_run = 0
        current_run = 0

        for is_missing in missing:
            if is_missing:
                current_run += 1
                longest_run = max(
                    longest_run, current_run
                )
            else:
                current_run = 0

        assert longest_run >= 10

    print("Missing interval test: PASS")


def test_missing_interval_engine_boundary():
    df = create_dummy_data()

    degraded = SensorDegradationEngine(
        random_seed=42
    ).missing_interval(
        df,
        sensors=["sensor_1"],
        interval_length=10,
        num_intervals=1,
    )

    for _, group in degraded.groupby(
        "engine_id", sort=False
    ):
        assert len(group) == 50
        assert int(group["sensor_1"].isna().sum()) == 10

    assert degraded["engine_id"].notna().all()
    print("Engine boundary test: PASS")


def test_invalid_missing_interval():
    df = create_dummy_data()
    engine = SensorDegradationEngine()

    invalid_cases = [
        {"interval_length": 0, "num_intervals": 1},
        {"interval_length": -1, "num_intervals": 1},
        {"interval_length": 10, "num_intervals": 0},
        {"interval_length": 10, "num_intervals": -1},
        {"interval_length": 51, "num_intervals": 1},
        {"interval_length": 10, "num_intervals": 6},
    ]

    for params in invalid_cases:
        try:
            engine.missing_interval(
                df, sensors=["sensor_1"], **params
            )
        except ValueError:
            continue

        raise AssertionError(
            f"Invalid interval parameters accepted: {params}"
        )

    print("Invalid missing interval test: PASS")


def test_missing_engine_id():
    df = create_dummy_data().drop(
        columns=["engine_id"]
    )

    try:
        SensorDegradationEngine().missing_interval(
            df,
            sensors=["sensor_1"],
            interval_length=10,
            num_intervals=1,
        )
    except ValueError:
        print("Missing engine_id test: PASS")
        return

    raise AssertionError(
        "missing_interval accepted data without engine_id."
    )


# ============================================================
# Additive Gaussian Noise Tests
# ============================================================

def test_add_noise():
    df = create_dummy_data()
    original = df.copy()

    degraded = SensorDegradationEngine(
        random_seed=42
    ).add_noise(
        df,
        sensors=["sensor_1"],
        noise_std=0.20,
    )

    pd.testing.assert_frame_equal(df, original)
    pd.testing.assert_series_equal(
        df["sensor_2"], degraded["sensor_2"]
    )

    assert len(degraded) == len(df)

    assert not np.allclose(
        df["sensor_1"].to_numpy(),
        degraded["sensor_1"].to_numpy(),
    )

    noise = degraded["sensor_1"] - df["sensor_1"]
    assert abs(float(noise.mean())) < 1.0

    print("Noise degradation test: PASS")


def test_invalid_noise():
    df = create_dummy_data()
    engine = SensorDegradationEngine()

    for invalid_value in (-0.1, np.nan, np.inf):
        try:
            engine.add_noise(
                df,
                sensors=["sensor_1"],
                noise_std=invalid_value,
            )
        except ValueError:
            continue

        raise AssertionError(
            f"Invalid noise_std {invalid_value!r} was accepted."
        )

    print("Invalid noise parameter test: PASS")


# ============================================================
# Persistent Bias Tests
# ============================================================

def test_positive_bias():
    df = create_dummy_data()
    original = df.copy()

    degraded = SensorDegradationEngine(
        random_seed=42
    ).add_bias(
        df,
        sensors=["sensor_1"],
        bias_scale=0.20,
        direction="positive",
    )

    pd.testing.assert_frame_equal(df, original)
    pd.testing.assert_series_equal(
        df["sensor_2"], degraded["sensor_2"]
    )

    observed_bias = (
        degraded["sensor_1"].to_numpy()
        - df["sensor_1"].to_numpy()
    )

    expected_bias = 0.20 * df["sensor_1"].std(ddof=0)

    assert np.all(observed_bias > 0)
    assert np.allclose(observed_bias, expected_bias)

    print("Positive bias test: PASS")


def test_negative_bias():
    df = create_dummy_data()

    degraded = SensorDegradationEngine(
        random_seed=42
    ).add_bias(
        df,
        sensors=["sensor_1"],
        bias_scale=0.20,
        direction="negative",
    )

    observed_bias = (
        degraded["sensor_1"].to_numpy()
        - df["sensor_1"].to_numpy()
    )

    expected_bias = -0.20 * df["sensor_1"].std(ddof=0)

    assert np.all(observed_bias < 0)
    assert np.allclose(observed_bias, expected_bias)

    print("Negative bias test: PASS")


def test_random_bias_reproducibility():
    df = create_dummy_data()
    sensors = ["sensor_1", "sensor_2"]

    result_1 = SensorDegradationEngine(
        random_seed=42
    ).add_bias(
        df,
        sensors=sensors,
        bias_scale=0.20,
        direction="random",
    )

    result_2 = SensorDegradationEngine(
        random_seed=42
    ).add_bias(
        df,
        sensors=sensors,
        bias_scale=0.20,
        direction="random",
    )

    pd.testing.assert_frame_equal(result_1, result_2)
    print("Random bias reproducibility test: PASS")


def test_invalid_bias():
    df = create_dummy_data()
    engine = SensorDegradationEngine()

    for invalid_scale in (-0.1, np.nan, np.inf):
        try:
            engine.add_bias(
                df,
                sensors=["sensor_1"],
                bias_scale=invalid_scale,
            )
        except ValueError:
            continue

        raise AssertionError(
            f"Invalid bias_scale {invalid_scale!r} was accepted."
        )

    try:
        engine.add_bias(
            df,
            sensors=["sensor_1"],
            bias_scale=0.20,
            direction="invalid",
        )
    except ValueError:
        print("Invalid bias parameter test: PASS")
        return

    raise AssertionError("Invalid bias direction was accepted.")


def test_bias_other_sensors_unchanged():
    df = create_dummy_data()

    degraded = SensorDegradationEngine(
        random_seed=42
    ).add_bias(
        df,
        sensors=["sensor_1"],
        bias_scale=0.20,
        direction="positive",
    )

    for sensor in ("sensor_2", "sensor_3", "sensor_4"):
        pd.testing.assert_series_equal(
            df[sensor], degraded[sensor]
        )

    print("Bias sensor isolation test: PASS")


def test_zero_bias():
    df = create_dummy_data()

    degraded = SensorDegradationEngine(
        random_seed=42
    ).add_bias(
        df,
        sensors=["sensor_1"],
        bias_scale=0.0,
        direction="positive",
    )

    pd.testing.assert_frame_equal(df, degraded)
    print("Zero bias test: PASS")


# ============================================================
# Progressive Drift Tests
# ============================================================

def test_positive_drift():
    df = create_dummy_data()
    original = df.copy()

    degraded = SensorDegradationEngine(
        random_seed=42
    ).add_drift(
        df,
        sensors=["sensor_1"],
        drift_scale=0.50,
        direction="positive",
    )

    pd.testing.assert_frame_equal(df, original)
    pd.testing.assert_series_equal(
        df["sensor_2"], degraded["sensor_2"]
    )

    for _, group in df.groupby(
        "engine_id", sort=False
    ):
        indices = group.index

        original_values = (
            df.loc[indices, "sensor_1"].to_numpy()
        )
        degraded_values = (
            degraded.loc[indices, "sensor_1"].to_numpy()
        )

        offsets = degraded_values - original_values
        expected_final = (
            0.50 * group["sensor_1"].std(ddof=0)
        )

        # Drift starts at zero.
        assert np.isclose(offsets[0], 0.0)

        # Drift never decreases.
        assert np.all(np.diff(offsets) >= -1e-12)

        # Final drift matches the requested scale.
        assert np.isclose(offsets[-1], expected_final)

    print("Positive progressive drift test: PASS")


def test_negative_drift():
    df = create_dummy_data()

    degraded = SensorDegradationEngine(
        random_seed=42
    ).add_drift(
        df,
        sensors=["sensor_1"],
        drift_scale=0.50,
        direction="negative",
    )

    for _, group in df.groupby(
        "engine_id", sort=False
    ):
        indices = group.index

        offsets = (
            degraded.loc[indices, "sensor_1"].to_numpy()
            - df.loc[indices, "sensor_1"].to_numpy()
        )

        expected_final = (
            -0.50 * group["sensor_1"].std(ddof=0)
        )

        assert np.isclose(offsets[0], 0.0)
        assert np.all(np.diff(offsets) <= 1e-12)
        assert np.isclose(offsets[-1], expected_final)

    print("Negative progressive drift test: PASS")


def test_random_drift_reproducibility():
    df = create_dummy_data()
    sensors = ["sensor_1", "sensor_2"]

    result_1 = SensorDegradationEngine(
        random_seed=42
    ).add_drift(
        df,
        sensors=sensors,
        drift_scale=0.30,
        direction="random",
    )

    result_2 = SensorDegradationEngine(
        random_seed=42
    ).add_drift(
        df,
        sensors=sensors,
        drift_scale=0.30,
        direction="random",
    )

    pd.testing.assert_frame_equal(result_1, result_2)
    print("Random drift reproducibility test: PASS")


def test_invalid_drift():
    df = create_dummy_data()
    engine = SensorDegradationEngine()

    for invalid_scale in (-0.1, np.nan, np.inf):
        try:
            engine.add_drift(
                df,
                sensors=["sensor_1"],
                drift_scale=invalid_scale,
            )
        except ValueError:
            continue

        raise AssertionError(
            f"Invalid drift_scale {invalid_scale!r} was accepted."
        )

    try:
        engine.add_drift(
            df,
            sensors=["sensor_1"],
            drift_scale=0.50,
            direction="invalid",
        )
    except ValueError:
        print("Invalid drift parameter test: PASS")
        return

    raise AssertionError("Invalid drift direction was accepted.")


def test_drift_requires_engine_id():
    df = create_dummy_data().drop(
        columns=["engine_id"]
    )

    try:
        SensorDegradationEngine().add_drift(
            df,
            sensors=["sensor_1"],
            drift_scale=0.50,
        )
    except ValueError:
        print("Drift engine_id validation test: PASS")
        return

    raise AssertionError(
        "add_drift accepted data without engine_id."
    )


def test_zero_drift():
    df = create_dummy_data()

    degraded = SensorDegradationEngine(
        random_seed=42
    ).add_drift(
        df,
        sensors=["sensor_1"],
        drift_scale=0.0,
        direction="positive",
    )

    pd.testing.assert_frame_equal(df, degraded)
    print("Zero drift test: PASS")


# ============================================================
# Run All Tests
# ============================================================

def main():
    print("=" * 70)
    print("SensorGuard - Sensor Degradation Tests")
    print("=" * 70)
    print()

    tests = [
        test_random_dropout,
        test_reproducibility,
        test_invalid_rate,
        test_invalid_sensor,
        test_missing_interval,
        test_missing_interval_engine_boundary,
        test_invalid_missing_interval,
        test_missing_engine_id,
        test_add_noise,
        test_invalid_noise,
        test_positive_bias,
        test_negative_bias,
        test_random_bias_reproducibility,
        test_invalid_bias,
        test_bias_other_sensors_unchanged,
        test_zero_bias,
        test_positive_drift,
        test_negative_drift,
        test_random_drift_reproducibility,
        test_invalid_drift,
        test_drift_requires_engine_id,
        test_zero_drift,
    ]

    for test in tests:
        test()

    print()
    print("=" * 70)
    print(f"All {len(tests)} degradation tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
