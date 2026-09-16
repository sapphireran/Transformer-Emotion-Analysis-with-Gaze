import numpy as np
import pytest

from gaze_emotion.scaling import (
    average_subject_tables,
    fill_missing,
    mean_normalize,
    min_max_scale,
    scale_features,
    standard_scale,
)


def test_min_max_range_and_constant_column():
    data = np.array([[0.0, 5.0], [10.0, 5.0], [5.0, 5.0]])
    scaled = min_max_scale(data)
    assert scaled[0, 0] == pytest.approx(0.0)
    assert scaled[1, 0] == pytest.approx(1.0)
    assert scaled[2, 0] == pytest.approx(0.5)
    assert np.allclose(scaled[:, 1], 0.0)


def test_standard_scale_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=3.0, scale=2.0, size=(200, 3))
    scaled = standard_scale(data)
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-12)


def test_mean_normalize_centers():
    data = np.array([[0.0], [10.0], [4.0]])
    scaled = mean_normalize(data)
    assert scaled.mean() == pytest.approx(0.0)
    assert scaled.max() - scaled.min() == pytest.approx(1.0)


def test_fill_missing_methods():
    data = np.array([[1.0, np.nan], [3.0, 2.0], [np.nan, 4.0]])
    zeros = fill_missing(data, "zeros")
    assert zeros[0, 1] == 0.0
    assert zeros[2, 0] == 0.0
    mean_filled = fill_missing(data, "mean")
    assert mean_filled[2, 0] == pytest.approx(2.0)
    min_filled = fill_missing(data, "min")
    assert min_filled[0, 1] == pytest.approx(2.0)


def test_treat_zero_as_missing():
    data = np.array([[0.0], [4.0], [8.0]])
    filled = fill_missing(data, "mean", treat_zero_as_missing=True)
    assert filled[0, 0] == pytest.approx(6.0)


def test_scale_features_dispatch_raw_is_copy():
    data = np.array([[1.0, 2.0]])
    raw = scale_features(data, "raw")
    raw[0, 0] = 99
    assert data[0, 0] == 1.0
    with pytest.raises(ValueError):
        scale_features(data, "log")


def test_average_subject_tables_nanmean():
    a = np.array([[1.0, 2.0], [3.0, np.nan]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    avg = average_subject_tables([a, b])
    assert avg[0, 0] == pytest.approx(3.0)
    assert avg[1, 1] == pytest.approx(8.0)


def test_average_empty_raises():
    with pytest.raises(ValueError):
        average_subject_tables([])
