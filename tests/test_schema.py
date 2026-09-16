from gazekit.schema import (
    CANONICAL_GAZE,
    EXPECTED_ROWS,
    SENTIMENT_FROM_INT,
    SENTIMENT_TO_INT,
    gaze_columns_in,
)


def test_label_maps_are_inverses():
    assert SENTIMENT_TO_INT["NEGATIVE"] == 0
    assert SENTIMENT_TO_INT["NEUTRAL"] == 1
    assert SENTIMENT_TO_INT["POSITIVE"] == 2
    assert {SENTIMENT_FROM_INT[i] for i in range(3)} == set(SENTIMENT_TO_INT)


def test_canonical_gaze_order_matches_training_scripts():
    # model_ZuCo_SST.py: df[['nFixations', 'FFD', 'GPT', 'TRT', 'GD']]
    assert CANONICAL_GAZE == ("nFixations", "FFD", "GPT", "TRT", "GD")


def test_gaze_columns_in_accepts_nfix_alias():
    assert gaze_columns_in(["nFix", "FFD", "GPT", "TRT", "GD"]) == list(CANONICAL_GAZE)
    assert gaze_columns_in(["sentence"]) == []


def test_expected_row_constants():
    assert EXPECTED_ROWS["zuco_sst_combined"] == 400
    assert EXPECTED_ROWS["zuco_sst_train"] + EXPECTED_ROWS["zuco_sst_valid"] + EXPECTED_ROWS[
        "zuco_sst_test"
    ] == EXPECTED_ROWS["zuco_sst_combined"]
