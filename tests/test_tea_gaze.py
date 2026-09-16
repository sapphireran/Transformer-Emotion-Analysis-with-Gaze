"""Tests for the personal example helpers. Uses the checked-in CSVs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tea_gaze.baselines import gaze_only_logistic_cv, majority_cv
from tea_gaze.features import (
    collinearity_pairs,
    compare_scaling,
    correlation_with_label,
    scale_numeric_frame,
    word_to_sentence_means,
)
from tea_gaze.io import (
    load_full_sst_splits,
    load_word_averages,
    load_zuco_combined,
    load_zuco_sentiment,
    load_zuco_splits,
)
from tea_gaze.metrics import classification_metrics, majority_baseline
from tea_gaze.models import ToyEyeTrackingFusion, hashed_bow_vector
from tea_gaze.paths import DataPaths
from tea_gaze.schema import (
    SST_MODEL_FEATURES,
    ZUCO_MODEL_FEATURES,
    label_name,
    required_columns,
)


def test_label_name_and_required_columns() -> None:
    assert label_name(0) == "NEGATIVE"
    assert label_name(2) == "POSITIVE"
    with pytest.raises(ValueError):
        label_name(9)
    assert "nFixations" in required_columns("zuco_sentence_et")
    assert "nFix" in required_columns("full_sst")


def test_zuco_combined_shape_and_labels() -> None:
    bundle = load_zuco_combined(scaling="standard")
    assert bundle.n_rows == 400
    assert set(bundle.frame["sentiment_label"].unique()) <= {0, 1, 2}
    counts = bundle.frame["sentiment_label"].value_counts().to_dict()
    assert counts[0] == 123
    assert counts[1] == 137
    assert counts[2] == 140


def test_full_sst_partition() -> None:
    splits = load_full_sst_splits()
    assert splits["combined"].n_rows == 11853
    assert splits["train"].n_rows + splits["valid"].n_rows + splits["test"].n_rows == 11853
    ids = {
        name: set(splits[name].frame["sentence_id"])
        for name in ("train", "valid", "test")
    }
    assert not (ids["train"] & ids["valid"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["valid"] & ids["test"])


def test_zuco_split_files_are_a_partition() -> None:
    combined = load_zuco_combined().frame
    parts = load_zuco_splits()
    assert sum(b.n_rows for b in parts.values()) == len(combined)
    ids = {name: set(bundle.frame["sentence_id"]) for name, bundle in parts.items()}
    assert not (ids["train"] & ids["valid"])
    assert ids["train"] | ids["valid"] | ids["test"] == set(combined["sentence_id"])


def test_metrics_and_majority() -> None:
    labels = np.array([0, 0, 1, 2, 2, 2])
    preds = np.array([0, 1, 1, 2, 2, 0])
    bundle = classification_metrics(preds, labels)
    assert 0.0 <= bundle.accuracy <= 1.0
    majority, metrics = majority_baseline(labels)
    assert majority == 2
    assert metrics.accuracy == pytest.approx(0.5)


def test_hashed_bow_is_stable_and_normalized() -> None:
    left = hashed_bow_vector("Beautifully crafted, engaging filmmaking")
    right = hashed_bow_vector("Beautifully crafted, engaging filmmaking")
    assert np.allclose(left, right)
    assert np.isclose(np.linalg.norm(left), 1.0)
    empty = hashed_bow_vector("...")
    assert empty.shape == left.shape
    assert np.allclose(empty, 0)


def test_toy_fusion_overfits_tiny_set() -> None:
    texts = [
        "terrible boring waste",
        "awful dull mess",
        "fine average movie",
        "okay regular film",
        "wonderful brilliant joy",
        "amazing delightful treat",
    ]
    labels = np.array([0, 0, 1, 1, 2, 2])
    et = np.array(
        [
            [2.0, 1.0, 1.5, 2.0, 1.0],
            [2.1, 1.1, 1.4, 2.2, 1.1],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, -0.1, 0.0, 0.1, 0.0],
            [-1.0, -0.5, -0.8, -1.0, -0.4],
            [-1.1, -0.4, -0.7, -0.9, -0.5],
        ]
    )
    model = ToyEyeTrackingFusion(seed=3, C=10.0)
    model.fit(texts, et, labels)
    preds = model.predict(texts, et)
    assert preds.shape == labels.shape
    # Tiny linearly-ish set; the toy model should at least be able to fit most of it.
    assert float((preds == labels).mean()) >= 0.5


def test_scaling_and_collinearity() -> None:
    frame = load_zuco_combined(scaling="minmax").frame
    scaled = scale_numeric_frame(frame, list(ZUCO_MODEL_FEATURES), "standard")
    assert abs(scaled["nFixations"].mean()) < 1e-6
    views = compare_scaling(frame, ZUCO_MODEL_FEATURES)
    assert set(views) == {"raw", "standard", "minmax"}
    sst = load_full_sst_splits()["combined"].frame
    pairs = collinearity_pairs(sst[list(SST_MODEL_FEATURES)].corr(), threshold=0.95)
    names = {(a, b) for a, b, _ in pairs}
    assert ("nFix", "FFD") in names or ("FFD", "nFix") in names


def test_word_to_sentence_keeps_400_sentences() -> None:
    words = load_word_averages().frame
    grouped = word_to_sentence_means(words)
    assert grouped["Sent_ID"].nunique() == 400
    assert "n_words" in grouped.columns
    assert grouped["n_words"].sum() == len(words)


def test_zuco_text_matches_combined_ids() -> None:
    text = load_zuco_sentiment().frame
    combined = load_zuco_combined().frame
    assert set(text["sentence_id"]) == set(combined["sentence_id"])
    assert list(text["sentence"]) == list(combined["sentence"])


def test_correlation_helper_matches_pandas() -> None:
    frame = load_zuco_combined().frame
    series = correlation_with_label(frame, ZUCO_MODEL_FEATURES)
    expected = frame[list(ZUCO_MODEL_FEATURES) + ["sentiment_label"]].corr()["sentiment_label"]
    for col in ZUCO_MODEL_FEATURES:
        assert series[col] == pytest.approx(expected[col])


def test_majority_cv_and_gaze_cv_run() -> None:
    frame = load_zuco_combined().frame
    maj = majority_cv(frame, n_splits=3, seed=0)
    gaze = gaze_only_logistic_cv(frame, ZUCO_MODEL_FEATURES, n_splits=3, seed=0)
    assert len(maj.folds) == 3
    assert 0.0 <= maj.mean.accuracy <= 1.0
    assert 0.0 <= gaze.mean.f1 <= 1.0


def test_paths_point_at_real_files() -> None:
    paths = DataPaths.from_cwd()
    assert paths.zuco_combined_standard.is_file()
    assert paths.full_sst_train.is_file()
    assert paths.subject_sentence_csv(3).is_file()
    with pytest.raises(ValueError):
        paths.subject_sentence_csv(0)
