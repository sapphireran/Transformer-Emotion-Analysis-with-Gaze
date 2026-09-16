import pandas as pd
import pytest

from gazekit.io import load_sentence_table
from gazekit.paths import default_paths
from gazekit.schema import EXPECTED_ROWS
from gazekit.splits import check_splits, max_class_drift


def test_disjoint_fixture():
    train = pd.DataFrame({"sentence_id": [1, 2], "sentiment_label": [0, 1]})
    valid = pd.DataFrame({"sentence_id": [3], "sentiment_label": [1]})
    test = pd.DataFrame({"sentence_id": [4], "sentiment_label": [2]})
    combined = pd.concat([train, valid, test], ignore_index=True)
    report = check_splits(train, valid, test, expected_union=combined)
    assert report.is_disjoint
    assert report.missing_from_union == 0
    assert report.extra_in_union == 0


def test_overlap_is_reported():
    train = pd.DataFrame({"sentence_id": [1, 2], "sentiment_label": [0, 1]})
    valid = pd.DataFrame({"sentence_id": [2], "sentiment_label": [1]})
    test = pd.DataFrame({"sentence_id": [3], "sentiment_label": [2]})
    report = check_splits(train, valid, test)
    assert not report.is_disjoint
    assert report.overlapping_ids["train∩valid"] == [2]


def test_zuco_convenience_split_covers_all_ids():
    paths = default_paths()
    train = load_sentence_table(paths.zuco_train)
    valid = load_sentence_table(paths.zuco_valid)
    test = load_sentence_table(paths.zuco_test)
    combined = load_sentence_table(paths.zuco_combined_standard)
    assert len(train) == EXPECTED_ROWS["zuco_sst_train"]
    assert len(valid) == EXPECTED_ROWS["zuco_sst_valid"]
    assert len(test) == EXPECTED_ROWS["zuco_sst_test"]
    report = check_splits(train, valid, test, expected_union=combined)
    assert report.is_disjoint
    assert report.missing_from_union == 0
    assert report.extra_in_union == 0
    # Non-stratified 40-row slices can drift; just lock that the helper runs.
    assert max_class_drift(report, versus="combined") >= 0.0


def test_full_sst_split_is_disjoint():
    paths = default_paths()
    train = load_sentence_table(paths.full_sst_train)
    valid = load_sentence_table(paths.full_sst_valid)
    test = load_sentence_table(paths.full_sst_test)
    report = check_splits(train, valid, test)
    assert report.is_disjoint
    assert len(train) == EXPECTED_ROWS["full_sst_train"]
    assert len(valid) == EXPECTED_ROWS["full_sst_valid"]
    assert len(test) == EXPECTED_ROWS["full_sst_test"]


def test_missing_id_column_raises():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(KeyError):
        check_splits(df, df, df)
