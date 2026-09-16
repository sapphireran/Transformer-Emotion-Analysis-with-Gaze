from __future__ import annotations

import pandas as pd
import pytest

from gaze_emotion_examples.io import load_dataset
from gaze_emotion_examples.profiles import (
    profile_frame,
    reconstruct_sentence,
    sentence_profile,
    top_words_by_measure,
)


def test_sentence_zero_profile_matches_checked_in_words():
    words = load_dataset("zuco_word_raw")
    profile = sentence_profile(words, "0_NR")
    assert len(profile) == 22
    assert profile[0].word == "presents"
    assert profile[-1].word == "decency"
    assert "failing" in reconstruct_sentence(profile)
    top = top_words_by_measure(profile, "gpt", k=1)
    assert top[0].word == "decency"
    frame = profile_frame(profile)
    assert list(frame["word"].head(3)) == ["presents", "a", "good"]


def test_sentence_profile_accepts_integer_id():
    words = load_dataset("zuco_word_raw")
    by_int = sentence_profile(words, 0)
    by_str = sentence_profile(words, "0_NR")
    assert reconstruct_sentence(by_int) == reconstruct_sentence(by_str)


def test_unknown_sentence_and_measure():
    words = pd.DataFrame(
        {
            "Sent_ID": ["9_NR"],
            "Word_ID": [0],
            "Word": ["hello"],
            "nFixations": [1.0],
            "FFD": [100.0],
            "GPT": [120.0],
            "TRT": [130.0],
            "GD": [110.0],
            "WordLen": [5],
        }
    )
    with pytest.raises(KeyError):
        sentence_profile(words, "99_NR")
    profile = sentence_profile(words, "9_NR")
    with pytest.raises(ValueError):
        top_words_by_measure(profile, "pupil")
