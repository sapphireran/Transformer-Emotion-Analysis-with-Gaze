from teag_examples.paths import repo_root
from teag_examples.schema import (
    FULL_SST_GAZE_COLS,
    FULL_SST_N_SENTENCES,
    LABEL_NAMES,
    SENTIMENT_FROM_STRING,
    SUBJECT_3_SKIP_ORIGINAL,
    ZUCO_MODEL_GAZE_COLS,
    ZUCO_N_SENTENCES,
    ZUCO_SUBJECT_3_N_SENTENCES,
)


def test_repo_root_finds_training_scripts():
    root = repo_root()
    assert (root / "model_full_SST.py").is_file()
    assert (root / "model_ZuCo_SST.py").is_file()


def test_label_maps_are_three_class():
    assert SENTIMENT_FROM_STRING["NEGATIVE"] == 0
    assert SENTIMENT_FROM_STRING["NEUTRAL"] == 1
    assert SENTIMENT_FROM_STRING["POSITIVE"] == 2
    assert set(LABEL_NAMES) == {0, 1, 2}


def test_fusion_feature_order_is_documented():
    assert FULL_SST_GAZE_COLS == ("nFix", "FFD", "GPT", "TRT", "GD")
    assert ZUCO_MODEL_GAZE_COLS == ("nFixations", "FFD", "GPT", "TRT", "GD")


def test_subject3_skip_list_length():
    assert len(SUBJECT_3_SKIP_ORIGINAL) == 101
    assert ZUCO_N_SENTENCES - len(set(SUBJECT_3_SKIP_ORIGINAL)) == ZUCO_SUBJECT_3_N_SENTENCES
    assert FULL_SST_N_SENTENCES == 11853
