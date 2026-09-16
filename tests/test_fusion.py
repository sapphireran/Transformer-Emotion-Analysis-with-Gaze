from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gaze_emotion_examples.fusion import GazeFusionForward, softmax, text_vs_gaze_cv


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs[1], np.full(3, 1.0 / 3.0))


def test_fusion_shapes_and_batch_mismatch():
    model = GazeFusionForward(text_dim=8, gaze_dim=5, hidden=4, num_labels=3, seed=1)
    text = np.ones((3, 8))
    gaze = np.ones((3, 5))
    logits = model.forward(text, gaze)
    assert logits.shape == (3, 3)
    assert model.project_gaze(gaze).shape == (3, 4)
    assert model.predict(text, gaze).shape == (3,)
    with pytest.raises(ValueError, match="batch size"):
        model.forward(text, np.ones((2, 5)))
    with pytest.raises(ValueError, match="gaze dim"):
        model.project_gaze(np.ones((3, 4)))


def test_fusion_accepts_single_row():
    model = GazeFusionForward(text_dim=8, gaze_dim=5, hidden=4, seed=2)
    logits = model.forward(np.zeros(8), np.zeros(5))
    assert logits.shape == (1, 3)


def test_text_vs_gaze_cv_runs_on_tiny_balanced_frame():
    sentences = [
        "a terrible dull movie",
        "awful and boring",
        "bad acting throughout",
        "fine ordinary film",
        "a regular okay story",
        "neither good nor bad",
        "wonderful joyful picture",
        "excellent delightful work",
        "amazing beautiful scenes",
        "a terrible dull sequel",
        "awful and loud",
        "bad writing throughout",
        "fine ordinary sequel",
        "a regular okay script",
        "neither sharp nor dull",
        "wonderful joyful ending",
        "excellent delightful cast",
        "amazing beautiful score",
    ]
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    gaze = np.linspace(-1.0, 1.0, len(sentences))
    frame = pd.DataFrame(
        {
            "sentence": sentences,
            "sentiment_label": labels,
            "nFixations": gaze,
            "GD": gaze * 0.5,
            "TRT": gaze * 1.2,
            "FFD": gaze * 0.3,
            "GPT": gaze * 0.8,
        }
    )
    results = text_vs_gaze_cv(
        frame,
        text_column="sentence",
        label_column="sentiment_label",
        gaze_columns=["nFixations", "GD", "TRT", "FFD", "GPT"],
        n_splits=3,
        seed=0,
    )
    names = [result.name for result in results]
    assert names == ["gaze_only", "text_only", "text_plus_gaze"]
    for result in results:
        assert len(result.folds) == 3
        assert 0.0 <= result.mean_accuracy <= 1.0
        assert not result.to_frame().empty
