import numpy as np
import pytest

from teag_examples.fusion import (
    FusionConfig,
    NumpyEyeTrackingFusion,
    cross_entropy,
    softmax,
)
from teag_examples.metrics import calculate_metrics


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = softmax(logits)
    assert probs.shape == (2, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs[1, 0] == pytest.approx(1 / 3)


def test_tiny_fusion_shapes_and_eval_deterministic():
    cfg = FusionConfig(hidden_size=4, n_gaze_features=2, gaze_hidden=3, n_labels=2, dropout=0.5)
    model = NumpyEyeTrackingFusion(cfg, seed=1)
    rng = np.random.default_rng(2)
    pooled = rng.normal(size=(5, 4))
    gaze = rng.normal(size=(5, 2))
    logits = model.forward(pooled, gaze, train=False)
    assert logits.shape == (5, 2)
    assert np.allclose(logits, model.forward(pooled, gaze, train=False))
    assert model.gaze_hidden(gaze).shape == (5, 3)
    assert model.concat_width == 7
    assert model.predict(pooled, gaze).shape == (5,)


def test_gaze_change_moves_logits():
    cfg = FusionConfig(hidden_size=8, n_gaze_features=5, gaze_hidden=4, n_labels=3)
    model = NumpyEyeTrackingFusion(cfg, seed=3)
    pooled = np.zeros((1, 8))
    a = model.forward(pooled, np.zeros((1, 5)))
    b = model.forward(pooled, np.ones((1, 5)))
    assert not np.allclose(a, b)


def test_train_dropout_is_stochastic_eval_is_not():
    cfg = FusionConfig(hidden_size=6, n_gaze_features=3, gaze_hidden=3, n_labels=3, dropout=0.5)
    model = NumpyEyeTrackingFusion(cfg, seed=4)
    pooled = np.ones((4, 6))
    gaze = np.ones((4, 3))
    d1 = model.forward(pooled, gaze, train=True, dropout_rng=np.random.default_rng(10))
    d2 = model.forward(pooled, gaze, train=True, dropout_rng=np.random.default_rng(11))
    assert np.max(np.abs(d1 - d2)) > 1e-8


def test_hand_computed_forward():
    cfg = FusionConfig(hidden_size=2, n_gaze_features=1, gaze_hidden=1, n_labels=2, dropout=0.0)
    model = NumpyEyeTrackingFusion(cfg, seed=0)
    model.W_gaze = np.array([[2.0]])
    model.b_gaze = np.array([0.5])
    model.W_cls = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    model.b_cls = np.array([0.0, 0.0])
    pooled = np.array([[1.0, 2.0]])
    gaze = np.array([[3.0]])
    logits = model.forward(pooled, gaze, train=False)
    assert logits.shape == (1, 2)
    assert logits[0, 0] == pytest.approx(7.5)
    assert logits[0, 1] == pytest.approx(-4.5)


def test_cross_entropy_uniform():
    logits = np.zeros((4, 3))
    labels = np.array([0, 1, 2, 0])
    # log(1/3)
    assert cross_entropy(logits, labels) == pytest.approx(-np.log(1 / 3))


def test_calculate_metrics_perfect():
    y = [0, 1, 2, 0]
    acc, p, r, f1 = calculate_metrics(y, y)
    assert acc == pytest.approx(1.0)
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)
