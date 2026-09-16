from pathlib import Path

import numpy as np
import pytest

from gaze_emotion.datasets import (
    KNOWN_TABLES,
    feature_matrix,
    infer_fusion_columns,
    known_table,
    load_and_summarize,
    load_csv_rows,
    resolve_data_path,
    split_overlap,
    summarize_rows,
)


def test_known_zuco_table_shape_and_labels():
    summary = load_and_summarize(known_table("zuco_combined_standard"))
    assert summary.n_rows == 400
    assert sum(summary.label_counts.values()) == 400
    assert summary.label_counts["POSITIVE"] == 140
    assert summary.label_counts["NEUTRAL"] == 137
    assert summary.label_counts["NEGATIVE"] == 123
    assert "nFixations" in summary.feature_means


def test_full_sst_train_counts():
    summary = load_and_summarize(known_table("full_sst_train"))
    assert summary.n_rows == 9482
    assert summary.label_counts["POSITIVE"] == 3939
    assert "nFix" in summary.feature_means
    assert "nFixations" not in summary.columns


def test_infer_fusion_columns():
    zuco = load_csv_rows(known_table("zuco_train"))
    sst = load_csv_rows(known_table("full_sst_valid"))
    assert infer_fusion_columns(zuco[0])[0] == "nFixations"
    assert infer_fusion_columns(sst[0])[0] == "nFix"
    with pytest.raises(KeyError):
        infer_fusion_columns(["sentence"])


def test_feature_matrix_and_missing_column():
    rows = load_csv_rows(known_table("zuco_test"))
    matrix = feature_matrix(rows, ("nFixations", "FFD"))
    assert matrix.shape == (len(rows), 2)
    assert np.isfinite(matrix).all()
    with pytest.raises(KeyError):
        feature_matrix(rows, ("not_a_column",))


def test_split_files_do_not_overlap():
    train = load_csv_rows(known_table("zuco_train"))
    valid = load_csv_rows(known_table("zuco_valid"))
    test = load_csv_rows(known_table("zuco_test"))
    assert not split_overlap(train, valid)
    assert not split_overlap(train, test)
    assert not split_overlap(valid, test)
    assert len(train) + len(valid) + len(test) == 400


def test_resolve_and_unknown_table():
    path = resolve_data_path("ZuCo_SST_data/test.csv")
    assert path.is_file()
    with pytest.raises(FileNotFoundError):
        resolve_data_path("does/not/exist.csv")
    with pytest.raises(KeyError):
        known_table("nope")
    assert set(KNOWN_TABLES)  # catalog is non-empty


def test_summarize_without_labels():
    rows = [{"nFixations": "1", "FFD": "2", "GPT": "3", "TRT": "4", "GD": "5"}]
    summary = summarize_rows(rows, path="memory")
    assert summary.n_rows == 1
    assert summary.label_counts == {}
    assert summary.feature_means["nFixations"] == 1.0


def test_repo_paths_are_absolute():
    assert Path(known_table("zuco_text_only")).is_absolute()
