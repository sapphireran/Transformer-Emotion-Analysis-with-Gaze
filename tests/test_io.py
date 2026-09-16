import pandas as pd
import pytest

from gazekit.io import (
    add_token_count,
    canonicalize_gaze,
    gaze_matrix,
    load_headerless_sst,
    load_sentence_table,
    load_word_table,
    require_gaze,
)
from gazekit.paths import default_paths
from gazekit.schema import CANONICAL_GAZE, EXPECTED_ROWS


def test_canonicalize_renames_nfix():
    df = pd.DataFrame({"nFix": [1.0], "FFD": [0.2]})
    out = canonicalize_gaze(df)
    assert "nFixations" in out.columns
    assert "nFix" not in out.columns
    assert out.loc[0, "nFixations"] == 1.0


def test_canonicalize_does_not_duplicate_when_both_exist():
    df = pd.DataFrame({"nFix": [1.0], "nFixations": [9.0], "FFD": [0.1]})
    out = canonicalize_gaze(df)
    assert "nFix" not in out.columns
    assert out.loc[0, "nFixations"] == 1.0


def test_add_token_count():
    df = pd.DataFrame({"sentence": ["one two three", ""]})
    out = add_token_count(df)
    assert list(out["n_tokens"]) == [3, 0]


def test_require_gaze_raises():
    with pytest.raises(KeyError):
        require_gaze(pd.DataFrame({"nFixations": [1]}))


def test_load_zuco_combined_real_csv():
    paths = default_paths()
    df = load_sentence_table(paths.zuco_combined_standard)
    assert len(df) == EXPECTED_ROWS["zuco_sst_combined"]
    assert all(col in df.columns for col in CANONICAL_GAZE)
    matrix = gaze_matrix(df)
    assert matrix.shape == (400, 5)
    assert df["sentiment_label"].isin({0, 1, 2}).all()
    assert "n_tokens" in df.columns


def test_load_full_sst_train_renames_nfix():
    paths = default_paths()
    df = load_sentence_table(paths.full_sst_train)
    assert len(df) == EXPECTED_ROWS["full_sst_train"]
    assert "nFixations" in df.columns
    assert "nFix" not in df.columns


def test_load_headerless_sst():
    paths = default_paths()
    df = load_headerless_sst(paths.full_sst_text)
    assert len(df) >= 1000
    assert set(df["sentiment_name"]) <= {"NEGATIVE", "NEUTRAL", "POSITIVE"}
    assert df["sentiment_label"].isin({0, 1, 2}).all()


def test_load_word_table_parses_sent_id():
    paths = default_paths()
    df = load_word_table(paths.word_averages)
    assert "sentence_id" in df.columns
    assert df["sentence_id"].min() == 0
    assert "word" in df.columns
    assert len(df) > 1000
