from __future__ import annotations

import pytest

from gaze_emotion_examples.catalog import DATASETS, get_dataset, list_datasets
from gaze_emotion_examples.io import load_dataset
from gaze_emotion_examples.paths import repo_root


def test_repo_root_finds_training_scripts():
    root = repo_root()
    assert (root / "model_ZuCo_SST.py").is_file()
    assert (root / "ZuCo_SST_data").is_dir()


def test_catalog_keys_are_unique():
    keys = [spec.key for spec in DATASETS]
    assert len(keys) == len(set(keys))


def test_get_dataset_unknown_key():
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_dataset("not-a-real-table")


def test_every_catalogued_file_exists_and_loads():
    for spec in DATASETS:
        assert spec.path.is_file(), spec.relative_path
        frame = load_dataset(spec)
        assert len(frame) > 0
        for column in spec.gaze_columns:
            assert column in frame.columns


def test_list_datasets_filters_kind():
    words = list_datasets("word")
    assert {spec.key for spec in words} == {"zuco_word_raw", "sst_word_predicted", "provo_word"}
