from __future__ import annotations

import pandas as pd
import pytest

from gaze_emotion_examples.io import load_dataset
from gaze_emotion_examples.splits import audit_split


def test_stored_zuco_and_sst_splits_are_clean():
    zuco = audit_split(
        "zuco",
        load_dataset("zuco_sst_standard"),
        load_dataset("zuco_sst_train"),
        load_dataset("zuco_sst_valid"),
        load_dataset("zuco_sst_test"),
        id_column="sentence_id",
        label_column="sentiment_label",
    )
    sst = audit_split(
        "sst",
        load_dataset("sst_full"),
        load_dataset("sst_train"),
        load_dataset("sst_valid"),
        load_dataset("sst_test"),
        id_column="sentence_id",
        label_column="sentiment_label",
    )
    assert zuco.ok
    assert sst.ok
    assert zuco.train_rows == 320
    assert zuco.valid_rows == 40
    assert zuco.test_rows == 40
    assert sst.train_rows + sst.valid_rows + sst.test_rows == 11853


def test_audit_detects_overlap():
    parent = pd.DataFrame({"id": [1, 2, 3], "sentiment_label": [0, 1, 2]})
    train = pd.DataFrame({"id": [1, 2], "sentiment_label": [0, 1]})
    valid = pd.DataFrame({"id": [2], "sentiment_label": [1]})
    test = pd.DataFrame({"id": [3], "sentiment_label": [2]})
    audit = audit_split("overlap", parent, train, valid, test, id_column="id", label_column="sentiment_label")
    assert audit.disjoint is False
    assert audit.ok is False


def test_audit_detects_missing_parent_id():
    parent = pd.DataFrame({"id": [1, 2, 3]})
    train = pd.DataFrame({"id": [1]})
    valid = pd.DataFrame({"id": [2]})
    test = pd.DataFrame({"id": [2]})
    audit = audit_split("missing", parent, train, valid, test, id_column="id")
    assert 3 in audit.missing_ids
    assert audit.covers_parent is False
