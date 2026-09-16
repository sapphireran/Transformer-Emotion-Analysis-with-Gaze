import numpy as np
import pandas as pd
import pytest

from gazekit.baselines import (
    fit_gaze_only,
    fit_length_only,
    fit_majority,
    residualize,
)
from gazekit.io import load_sentence_table
from gazekit.metrics import calculate_metrics
from gazekit.paths import default_paths


def test_calculate_metrics_perfect():
    m = calculate_metrics([0, 1, 2], [0, 1, 2])
    assert m.accuracy == 1.0
    assert m.f1_macro == 1.0
    assert m.n == 3


def test_calculate_metrics_handles_missing_class():
    m = calculate_metrics([0, 0, 0], [0, 1, 2])
    assert m.accuracy == pytest.approx(1 / 3)
    assert m.precision_macro >= 0.0


def test_majority_on_imbalanced_fixture():
    df = pd.DataFrame(
        {
            "sentiment_label": [2] * 8 + [0] * 2,
            "n_tokens": np.arange(10),
            "nFixations": np.linspace(-1, 1, 10),
            "FFD": np.linspace(0, 1, 10),
            "GPT": np.linspace(1, 0, 10),
            "TRT": np.linspace(-0.5, 0.5, 10),
            "GD": np.linspace(0.2, 0.8, 10),
        }
    )
    report = fit_majority(df, n_splits=2, seed=0)
    assert report.mean_accuracy == pytest.approx(0.8, abs=0.15)


def test_residualize_kills_linear_length_effect():
    rng = np.random.default_rng(1)
    n = 80
    length = rng.uniform(5, 40, n)
    noise = rng.normal(0, 0.05, n)
    df = pd.DataFrame(
        {
            "n_tokens": length,
            "nFixations": 0.3 * length + noise,
            "FFD": rng.normal(0, 1, n),
            "GPT": rng.normal(0, 1, n),
            "TRT": rng.normal(0, 1, n),
            "GD": rng.normal(0, 1, n),
            "sentiment_label": (length > 22).astype(int),
        }
    )
    resid = residualize(df, ["nFixations"], on="n_tokens")
    # Residual should be nearly uncorrelated with length.
    corr = np.corrcoef(resid["n_tokens"], resid["nFixations"])[0, 1]
    assert abs(corr) < 0.05


def test_real_zuco_baselines_run_and_majority_is_floor():
    df = load_sentence_table(default_paths().zuco_combined_standard)
    maj = fit_majority(df, n_splits=5, seed=42)
    gaze = fit_gaze_only(df, n_splits=5, seed=42)
    length = fit_length_only(df, n_splits=5, seed=42)
    assert 0.2 <= maj.mean_accuracy <= 0.8
    assert 0.0 <= gaze.mean_accuracy <= 1.0
    assert 0.0 <= length.mean_accuracy <= 1.0
    # All reports expose the dict used in docs / notebooks.
    assert "mean_f1_macro" in gaze.as_dict()
