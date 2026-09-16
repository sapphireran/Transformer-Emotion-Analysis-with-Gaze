import numpy as np
import pytest
from sklearn.metrics import accuracy_score as sk_acc
from sklearn.metrics import f1_score, precision_score, recall_score

from gaze_emotion.metrics import classification_report, confusion_matrix, weighted_scores


def test_weighted_scores_match_sklearn():
    y_true = np.array([0, 0, 1, 1, 2, 2, 2, 1, 0, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 2, 1, 0, 2])
    ours = weighted_scores(y_true, y_pred)
    assert ours["accuracy"] == pytest.approx(sk_acc(y_true, y_pred))
    assert ours["precision"] == pytest.approx(
        precision_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    assert ours["recall"] == pytest.approx(
        recall_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    assert ours["f1"] == pytest.approx(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def test_perfect_predictions():
    labels = [0, 1, 2, 1, 0]
    scores = weighted_scores(labels, labels)
    assert scores["accuracy"] == 1.0
    assert scores["f1"] == 1.0


def test_all_wrong_still_defined():
    scores = weighted_scores([0, 1, 2], [1, 2, 0])
    assert scores["accuracy"] == 0.0
    assert 0.0 <= scores["f1"] <= 1.0


def test_confusion_matrix_shape_and_counts():
    matrix = confusion_matrix([0, 1, 1, 2], [0, 1, 0, 2], num_labels=3)
    assert matrix.shape == (3, 3)
    assert matrix[1, 0] == 1
    assert matrix.sum() == 4


def test_classification_report_contains_labels():
    report = classification_report([0, 1, 2], [0, 1, 1])
    assert "NEGATIVE" in report
    assert "weighted" in report
    assert "accuracy:" in report


def test_empty_inputs():
    scores = weighted_scores([], [])
    assert scores["accuracy"] == 0.0
