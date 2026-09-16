import numpy as np
import pandas as pd

from teag_examples.stats import frame_profile, label_counts, profiles_to_markdown


def test_label_counts_sorted_ints():
    s = pd.Series([2, 0, 1, 1, 0])
    assert label_counts(s) == {0: 2, 1: 2, 2: 1}


def test_frame_profile_includes_sentence_stats():
    df = pd.DataFrame(
        {
            "sentence_id": [0, 1],
            "sentence": ["short", "a bit longer"],
            "sentiment_label": [0, 2],
            "nFix": [0.1, -0.2],
        }
    )
    profile = frame_profile(df, "toy")
    assert profile["rows"] == 2
    assert profile["n_missing"] == 0
    assert profile["label_counts"] == {0: 1, 2: 1}
    assert profile["sentence_chars"]["min"] == 5
    assert "nFix" in profile["moments"]
    md = profiles_to_markdown([profile])
    assert "`toy`" in md
    assert "1 / — / 1" in md


def test_moments_skip_id_columns():
    df = pd.DataFrame({"sentence_id": [1, 2, 3], "GD": [1.0, 2.0, 3.0]})
    profile = frame_profile(df, "ids")
    assert "sentence_id" not in profile["moments"]
    assert np.isclose(profile["moments"]["GD"]["mean"], 2.0)
