import pandas as pd

from gazekit.features import (
    class_balance,
    correlation_matrix,
    gaze_by_sentiment,
    majority_fraction,
    missingness_report,
    pairwise_mean_abs_diff,
    summarize_numeric,
    word_length_effects,
)
from gazekit.io import load_sentence_table, load_word_table
from gazekit.paths import default_paths
from gazekit.schema import CANONICAL_GAZE


def test_class_balance_and_majority_on_fixture():
    df = pd.DataFrame({"sentiment_label": [0, 0, 1, 2, 2, 2]})
    bal = class_balance(df)
    assert bal.loc[bal["sentiment_label"] == 2, "count"].iloc[0] == 3
    assert majority_fraction(df) == 0.5


def test_missingness_flags_absent_column():
    df = pd.DataFrame({"nFixations": [1.0, None]})
    report = missingness_report(df, ["nFixations", "FFD"])
    by = report.set_index("column")
    assert by.loc["nFixations", "n_missing"] == 1
    assert by.loc["FFD", "present"] is False


def test_summarize_numeric_skips_text():
    df = pd.DataFrame({"sentence": ["a"], "nFixations": [1.5]})
    out = summarize_numeric(df)
    assert list(out["column"]) == ["nFixations"]


def test_correlation_and_class_means_on_real_zuco():
    df = load_sentence_table(default_paths().zuco_combined_standard)
    corr = correlation_matrix(df)
    assert corr.shape == (5, 5)
    assert corr.loc["nFixations", "nFixations"] == pytest_approx_one()
    # TRT and nFixations should move together on real reading data.
    assert corr.loc["TRT", "nFixations"] > 0.3
    by = gaze_by_sentiment(df)
    assert set(by["sentiment_label"]) <= {0, 1, 2}
    diffs = pairwise_mean_abs_diff(df)
    assert not diffs.empty
    assert set(diffs["channel"]) == set(CANONICAL_GAZE)


def pytest_approx_one():
    return 1.0


def test_word_length_effects_monotone_ish():
    words = load_word_table(default_paths().word_averages)
    effects = word_length_effects(words)
    assert "len_bin" in effects.columns
    # Longer bins should not have *lower* nFixations than the shortest bin
    # by a wide margin — a loose sanity check on the export.
    nfix = effects.set_index("len_bin")["nFixations"]
    if "1-2" in nfix.index and "7-8" in nfix.index:
        assert nfix.loc["7-8"] >= nfix.loc["1-2"] * 0.8
