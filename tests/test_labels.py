from __future__ import annotations

import pandas as pd
import pytest

from gaze_emotion_examples.labels import (
    label_code,
    label_counts,
    label_name,
    majority_accuracy,
)


def test_label_roundtrip():
    assert label_name(0) == "negative"
    assert label_name(1) == "neutral"
    assert label_name(2) == "positive"
    assert label_code("NEGATIVE") == 0
    assert label_code("neutral") == 1
    assert label_code("POSITIVE") == 2


def test_label_name_rejects_unknown():
    with pytest.raises(ValueError):
        label_name(7)


def test_label_counts_and_majority():
    series = pd.Series([0, 0, 1, 2, 2, 2])
    table = label_counts(series)
    assert list(table["count"]) == [2, 1, 3]
    assert table.loc[table["name"] == "positive", "percent"].iloc[0] == pytest.approx(50.0)
    assert majority_accuracy(series) == pytest.approx(0.5)
    assert majority_accuracy(pd.Series(dtype=int)) == 0.0
