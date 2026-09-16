import numpy as np
import pytest

from gazekit.fusion import (
    ENCODER_HIDDEN,
    FUSED_WIDTH,
    GAZE_HIDDEN,
    GAZE_IN,
    NUM_LABELS,
    GazeFusionForward,
    assert_architecture_constants,
    shapes_report,
)
from gazekit.io import gaze_matrix, load_sentence_table
from gazekit.paths import default_paths


def test_architecture_constants():
    assert_architecture_constants()
    assert ENCODER_HIDDEN + GAZE_HIDDEN == FUSED_WIDTH


def test_shapes_report():
    shapes = shapes_report(batch_size=3, seed=1)
    assert shapes["pooled"] == (3, 768)
    assert shapes["eye_tracking_features"] == (3, 5)
    assert shapes["gaze_hidden"] == (3, 16)
    assert shapes["fused"] == (3, 784)
    assert shapes["logits"] == (3, 3)


def test_forward_rejects_wrong_et_width():
    model = GazeFusionForward(seed=0)
    pooled = np.zeros((2, 768))
    with pytest.raises(ValueError):
        model.project_gaze(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        model.fuse(np.zeros((2, 32)), np.zeros((2, 5)))
    with pytest.raises(ValueError):
        model.fuse(pooled, np.zeros((3, 5)))


def test_activated_projection_is_nonnegative():
    rng = np.random.default_rng(2)
    et = rng.normal(0, 3, (16, GAZE_IN))
    hidden = GazeFusionForward(seed=2, activated=True).project_gaze(et)
    assert hidden.shape == (16, GAZE_HIDDEN)
    assert np.all(hidden >= 0)


def test_real_et_batch_produces_finite_logits():
    df = load_sentence_table(default_paths().zuco_combined_standard)
    et = gaze_matrix(df)[:6]
    pooled = np.zeros((6, ENCODER_HIDDEN))
    logits = GazeFusionForward(seed=0)(pooled, et)
    assert logits.shape == (6, NUM_LABELS)
    assert np.isfinite(logits).all()


def test_single_row_input_is_accepted():
    model = GazeFusionForward(seed=0)
    logits = model(np.zeros(768), np.zeros(5))
    assert logits.shape == (1, 3)
