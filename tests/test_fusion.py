import numpy as np
import pytest

from gaze_emotion.fusion import (
    GazeFusionClassifier,
    cross_entropy,
    make_separable_batch,
    softmax,
)
from gaze_emotion.metrics import weighted_scores


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs[1], np.full(3, 1 / 3))


def test_cross_entropy_prefers_correct_class():
    labels = np.array([2, 2])
    good = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]])
    bad = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    assert cross_entropy(good, labels) < cross_entropy(bad, labels)


def test_forward_shapes():
    model = GazeFusionClassifier(text_dim=8, rng=np.random.default_rng(0))
    text = np.zeros((4, 8))
    gaze = np.zeros((4, 5))
    logits = model.forward(text, gaze)
    assert logits.shape == (4, 3)
    assert model.predict(text, gaze).shape == (4,)
    assert model.shapes.combined_dim == 8 + 16


def test_step_decreases_loss_on_separable_data():
    text, gaze, labels = make_separable_batch(n=64, text_dim=12, seed=3)
    model = GazeFusionClassifier(text_dim=12, rng=np.random.default_rng(3))
    first = model.step(text, gaze, labels, lr=0.2)
    for _ in range(40):
        last = model.step(text, gaze, labels, lr=0.2)
    assert last < first
    scores = weighted_scores(labels, model.predict(text, gaze))
    assert scores["accuracy"] >= 0.85


def test_one_sample_forward_accepts_vectors():
    model = GazeFusionClassifier(text_dim=6, rng=np.random.default_rng(2))
    logits = model.forward(np.ones(6), np.ones(5))
    assert logits.shape == (1, 3)
