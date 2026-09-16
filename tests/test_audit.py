import numpy as np
import pytest

from gaze_emotion.audit import (
    audit_splits,
    label_entropy,
    majority_baseline,
    subject_coverage_notes,
    subject_sentence_counts,
)
from gaze_emotion.constants import ZUCO_TASK1_EXPECTED_SENTENCES


def test_zuco_holdout_has_no_leakage_and_expected_sizes():
    audit = audit_splits(
        "ZuCo_SST_data/train.csv",
        "ZuCo_SST_data/valid.csv",
        "ZuCo_SST_data/test.csv",
    )
    assert audit.train_rows == 320
    assert audit.valid_rows == 40
    assert audit.test_rows == 40
    assert not audit.has_leakage


def test_full_sst_holdout_has_no_leakage():
    audit = audit_splits(
        "SST_data/train_full_sst.csv",
        "SST_data/valid_full_sst.csv",
        "SST_data/test_full_sst.csv",
    )
    assert audit.train_rows == 9482
    assert audit.valid_rows == 1185
    assert audit.test_rows == 1186
    assert not audit.has_leakage


def test_subject_3_is_short():
    counts = subject_sentence_counts()
    assert counts["1_SR.csv"] == ZUCO_TASK1_EXPECTED_SENTENCES
    assert counts["3_SR.csv"] == 299
    notes = subject_coverage_notes(counts)
    assert any("3_SR.csv" in note for note in notes)


def test_entropy_and_majority():
    balanced = {"NEGATIVE": 10, "NEUTRAL": 10, "POSITIVE": 10}
    skewed = {"NEGATIVE": 0, "NEUTRAL": 0, "POSITIVE": 30}
    assert label_entropy(balanced) == pytest.approx(np.log2(3), abs=1e-6)
    assert majority_baseline(skewed) == 1.0
    assert majority_baseline({}) == 0.0
