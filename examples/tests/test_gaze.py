import numpy as np
import pandas as pd
import pytest

from teag_examples.gaze import (
    almost_standardized,
    collinear_pairs,
    feature_label_correlations,
    in_unit_interval,
    pearson_matrix,
)
from teag_examples.io import load_full_sst_combined, load_zuco_combined
from teag_examples.schema import FULL_SST_GAZE_COLS, ZUCO_JOIN_GAZE_COLS


def test_pearson_diagonal_is_one():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 5.0]})
    corr = pearson_matrix(df, ["a", "b"])
    assert np.allclose(np.diag(corr), 1.0)
    assert corr.loc["a", "b"] == pytest.approx(corr.loc["b", "a"])


def test_collinear_pairs_threshold():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    df = pd.DataFrame({"x": x, "y": x + 0.001 * rng.normal(size=50), "z": rng.normal(size=50)})
    pairs = collinear_pairs(pearson_matrix(df, ["x", "y", "z"]), threshold=0.99)
    names = {(a, b) for a, b, _ in pairs}
    assert ("x", "y") in names
    assert ("x", "z") not in names


def test_zuco_standard_and_minmax_scaling():
    std = load_zuco_combined("standard")
    mm = load_zuco_combined("min_max")
    for col in ZUCO_JOIN_GAZE_COLS:
        assert almost_standardized(std[col], mean_tol=1e-6, std_tol=0.03)
        assert in_unit_interval(mm[col])


def test_sst_predicted_gaze_is_collinear_and_weakly_labeled():
    sst = load_full_sst_combined()
    corr = pearson_matrix(sst, FULL_SST_GAZE_COLS)
    pairs = collinear_pairs(corr, threshold=0.98)
    involved = {name for a, b, _ in pairs for name in (a, b)}
    assert {"nFix", "FFD", "GPT", "TRT"} <= involved
    label_r = feature_label_correlations(sst, FULL_SST_GAZE_COLS)
    assert float(label_r.abs().max()) < 0.1
