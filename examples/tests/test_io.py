from teag_examples.io import (
    load_full_sst_combined,
    load_full_sst_split,
    load_provo,
    load_raw_sst_sentences,
    load_sentence_average,
    load_subject_sentence_gaze,
    load_word_averages,
    load_zuco_combined,
    load_zuco_text,
)
from teag_examples.schema import (
    FULL_SST_LABEL_COUNTS,
    FULL_SST_N_SENTENCES,
    PROVO_ROWS,
    WORD_AVERAGES_ROWS,
    ZUCO_LABEL_COUNTS,
    ZUCO_N_SENTENCES,
    ZUCO_SUBJECT_3_N_SENTENCES,
)


def test_zuco_text_and_combined_row_counts():
    text = load_zuco_text()
    std = load_zuco_combined("standard")
    mm = load_zuco_combined("min_max")
    assert len(text) == ZUCO_N_SENTENCES
    assert len(std) == len(mm) == ZUCO_N_SENTENCES
    assert text["sentiment_label"].value_counts().sort_index().to_dict() == dict(
        ZUCO_LABEL_COUNTS
    )
    assert std.isna().sum().sum() == 0


def test_full_sst_counts_match_raw_strings():
    combined = load_full_sst_combined()
    raw = load_raw_sst_sentences()
    assert len(combined) == FULL_SST_N_SENTENCES
    assert len(raw) == FULL_SST_N_SENTENCES
    mapped = raw["label_string"].map({"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2})
    assert mapped.value_counts().sort_index().to_dict() == dict(FULL_SST_LABEL_COUNTS)
    assert combined["sentiment_label"].value_counts().sort_index().to_dict() == dict(
        FULL_SST_LABEL_COUNTS
    )


def test_subject_three_is_shorter():
    assert len(load_subject_sentence_gaze(1)) == ZUCO_N_SENTENCES
    assert len(load_subject_sentence_gaze(3)) == ZUCO_SUBJECT_3_N_SENTENCES
    assert len(load_subject_sentence_gaze(12)) == ZUCO_N_SENTENCES


def test_averages_and_word_tables_exist():
    raw = load_sentence_average("raw")
    assert len(raw) == ZUCO_N_SENTENCES
    words = load_word_averages()
    assert len(words) == WORD_AVERAGES_ROWS
    assert words["Sent_ID"].nunique() == ZUCO_N_SENTENCES
    provo = load_provo()
    assert len(provo) == PROVO_ROWS


def test_full_sst_split_files_load():
    assert len(load_full_sst_split("train")) == 9482
    assert len(load_full_sst_split("valid")) == 1185
    assert len(load_full_sst_split("test")) == 1186
