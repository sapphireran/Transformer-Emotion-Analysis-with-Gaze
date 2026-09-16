from __future__ import annotations

import pandas as pd

from gaze_emotion_examples.stats import correlation_matrix, describe_numeric, high_correlations


def test_describe_numeric_includes_missing():
    frame = pd.DataFrame({"a": [1.0, 2.0, None], "b": ["x", "y", "z"]})
    desc = describe_numeric(frame, ["a"])
    assert desc.loc[0, "column"] == "a"
    assert desc.loc[0, "missing"] == 1
    assert desc.loc[0, "mean"] == 1.5


def test_high_correlations_threshold():
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 2.0, 3.0],
            "z": [3.0, 2.0, 1.0, 0.0],
        }
    )
    corr = correlation_matrix(frame, ["x", "y", "z"])
    pairs = high_correlations(corr, threshold=0.99)
    labels = {frozenset((row.left, row.right)) for row in pairs.itertuples()}
    assert frozenset(("x", "y")) in labels
    assert frozenset(("x", "z")) in labels
