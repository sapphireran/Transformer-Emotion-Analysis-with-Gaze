import pandas as pd

from examples.lib.loaders import (
    FUSION_GAZE_COLUMNS,
    fusion_frame,
    label_counts,
    load_csv,
    load_headerless_sst,
    load_zuco_experiment,
    sentence_text,
    word_rows_for_sentence,
)


def test_zuco_experiment_shapes():
    std = load_zuco_experiment("standard")
    mm = load_zuco_experiment("minmax")
    assert len(std) == 400
    assert len(mm) == 400
    assert std["sentence_id"].tolist() == mm["sentence_id"].tolist()
    assert std["sentence"].tolist() == mm["sentence"].tolist()
    assert set(std["sentiment_label"].unique()) == {0, 1, 2}


def test_fusion_frame_renames_sst_columns():
    sst = load_csv("sst_train")
    gaze = fusion_frame(sst)
    assert list(gaze.columns) == list(FUSION_GAZE_COLUMNS)
    assert len(gaze) == len(sst)
    assert gaze.isna().sum().sum() == 0


def test_headerless_sst():
    df = load_headerless_sst()
    assert "sentence" in df.columns
    assert "sentiment_label" in df.columns
    assert len(df) == 11853
    assert set(df["polarity"].unique()) <= {"NEGATIVE", "NEUTRAL", "POSITIVE"}


def test_label_counts_sum():
    std = load_zuco_experiment("standard")
    counts = label_counts(std["sentiment_label"])
    assert counts == {0: 123, 1: 137, 2: 140}


def test_sentence_three_walkthrough_rows():
    text, label = sentence_text(3)
    assert "hilarious" in text.lower()
    assert label == 1
    words = word_rows_for_sentence(3, source="average")
    assert list(words["Word"].str.lower()) == [
        "slow",
        "silly",
        "and",
        "unintentionally",
        "hilarious",
    ]
    # The 15-letter adverb should be the stickiest TRT on the average.
    trt = pd.to_numeric(words["TRT"])
    assert words.loc[trt.idxmax(), "Word"].lower() == "unintentionally"


def test_subject_skip_on_sentence_three():
    s2 = word_rows_for_sentence(3, source="subject", subject=2)
    slow = s2.loc[s2["Word"].str.lower() == "slow"].iloc[0]
    assert float(slow["nFixations"]) == 0.0
