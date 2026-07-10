import numpy as np
import pytest

from deeplc.calibration import CalibrationError, PiecewiseLinearCalibration


def test_transform_requires_fit():
    cal = PiecewiseLinearCalibration()
    with pytest.raises(CalibrationError):
        cal.transform(np.array([0.0, 1.0], dtype=np.float32))


def test_piecewise_maps_predicted_to_measured_linear_case():
    # Linear relationship with no noise
    predicted = np.linspace(0.0, 10.0, 201, dtype=np.float32)
    measured = 2.0 * predicted + 5.0

    cal = PiecewiseLinearCalibration(number_of_splits=25, use_median=True)
    cal.fit(target=measured, source=predicted)

    transformed = cal.transform(predicted)
    # Should match measured closely
    assert transformed.shape == measured.shape
    np.testing.assert_allclose(transformed, measured, rtol=1e-5, atol=1e-4)


def test_piecewise_out_of_range_values_are_handled():
    predicted = np.linspace(0.0, 10.0, 201, dtype=np.float32)
    measured = 1.5 * predicted - 3.0

    cal = PiecewiseLinearCalibration(number_of_splits=20, use_median=True)
    cal.fit(target=measured, source=predicted)

    # Values outside the training range
    tr = np.array([-5.0, -0.1, 0.0, 5.0, 10.0, 10.1, 15.0], dtype=np.float32)
    out = cal.transform(tr)

    assert out.shape == tr.shape
    assert np.all(np.isfinite(out))
    # Monotonic transformation expected for linear calibration
    assert np.all(np.diff(out) >= -1e-6)


def test_breakpoint_transform_matches_linear_truth():
    x = np.linspace(0.0, 100.0, 1001, dtype=np.float32)
    y = 0.5 * x + 10.0
    cal = PiecewiseLinearCalibration(number_of_splits=40)
    cal.fit(target=y, source=x)
    out = cal.transform(x)
    np.testing.assert_allclose(out, y, rtol=1e-5, atol=1e-4)


def test_extrapolate_false_clips_inputs():
    x = np.linspace(10.0, 20.0, 101, dtype=np.float32)
    y = 3.0 * x - 7.0
    cal = PiecewiseLinearCalibration(number_of_splits=10, extrapolate=False)
    cal.fit(target=y, source=x)

    x_query = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 30.0], dtype=np.float32)
    out = cal.transform(x_query)

    # Values below clip to left bound; above to right bound
    left_val = (3.0 * 10.0) - 7.0
    right_val = (3.0 * 20.0) - 7.0
    assert np.isclose(out[0], left_val, rtol=1e-6)
    assert np.isclose(out[1], left_val, rtol=1e-6)
    assert np.isclose(out[-1], right_val, rtol=1e-6)


def test_zero_range_predicted_raises():
    import pytest

    from deeplc.calibration import CalibrationError

    x = np.ones(50, dtype=np.float32) * 7.5
    y = 2.0 * x + 1.0
    cal = PiecewiseLinearCalibration(number_of_splits=10)
    with pytest.raises(CalibrationError):
        cal.fit(target=y, source=x)


def test_piecewise_skips_sparse_segments_with_min_samples_threshold():
    source_dense = np.linspace(0.0, 80.0, 1000, dtype=np.float32)
    source_sparse = np.array([95.0, 97.0, 99.0], dtype=np.float32)
    source = np.concatenate([source_dense, source_sparse]).astype(np.float32)
    target = (1.2 * source) + 3.0

    cal_no_threshold = PiecewiseLinearCalibration(
        number_of_splits=100,
        min_samples_per_segment=1,
    )
    cal_no_threshold.fit(target=target, source=source)
    x_no_threshold, _ = cal_no_threshold.get_calibration_curve()

    cal_threshold = PiecewiseLinearCalibration(
        number_of_splits=100,
        min_samples_per_segment=10,
    )
    cal_threshold.fit(target=target, source=source)
    x_threshold, _ = cal_threshold.get_calibration_curve()

    assert x_threshold.size > 1
    assert x_threshold.size < x_no_threshold.size
