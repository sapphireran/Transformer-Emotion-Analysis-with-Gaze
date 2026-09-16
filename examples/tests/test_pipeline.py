from teag_examples.pipeline import max_abs_diff, reconstruct_zuco_combined
from teag_examples.schema import ZUCO_JOIN_GAZE_COLS, ZUCO_N_SENTENCES


def test_reconstructed_standard_matches_committed():
    reconstructed, committed = reconstruct_zuco_combined("standard")
    assert len(reconstructed) == len(committed) == ZUCO_N_SENTENCES
    assert max_abs_diff(reconstructed, committed, ZUCO_JOIN_GAZE_COLS) == 0.0
    left = reconstructed.sort_values("sentence_id")
    right = committed.sort_values("sentence_id")
    assert list(left["sentence"]) == list(right["sentence"])
    assert list(left["sentiment_label"]) == list(right["sentiment_label"])


def test_reconstructed_minmax_matches_committed():
    reconstructed, committed = reconstruct_zuco_combined("min_max")
    assert max_abs_diff(reconstructed, committed, ZUCO_JOIN_GAZE_COLS) == 0.0
    assert "SentLen" not in reconstructed.columns
    assert "SentLen" not in committed.columns
