import numpy as np

from examples.lib.metrics import majority_baseline, score_predictions


def test_perfect_predictions():
    y = np.array([0, 1, 2, 1, 0])
    bundle = score_predictions(y, y)
    assert bundle.accuracy == 1.0
    assert bundle.f1_macro == 1.0
    assert bundle.n == 5


def test_majority_baseline_on_known_prior():
    y = np.array([0, 0, 0, 1, 2])
    bundle = majority_baseline(y)
    assert bundle.accuracy == 0.6
    # Predicting all zeros: class 1 and 2 contribute 0 to macro F1.
    assert bundle.f1_macro < bundle.accuracy


def test_shape_mismatch():
    try:
        score_predictions([0, 1], [0])
    except ValueError:
        return
    raise AssertionError("expected ValueError")
