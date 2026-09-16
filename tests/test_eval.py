from tea_gaze.eval import confusion_matrix, mean_metrics, weighted_metrics
from tea_gaze.stats import pearson


def test_perfect_predictions_score_one():
    y = [0, 1, 2, 2, 1, 0]
    metrics = weighted_metrics(y, y)
    assert metrics.accuracy == 1
    assert metrics.f1 == 1
    assert metrics.support == 6


def test_confusion_matrix_shape():
    matrix = confusion_matrix([0, 1, 2], [0, 2, 2])
    assert matrix == [
        [1, 0, 0],
        [0, 0, 1],
        [0, 0, 1],
    ]


def test_mean_metrics_averages_folds():
    a = weighted_metrics([0, 1], [0, 1])
    b = weighted_metrics([0, 1], [1, 1])
    averaged = mean_metrics([a, b])
    assert 0.5 <= averaged.accuracy <= 1.0


def test_pearson_identical_series():
    assert pearson([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) == 1.0
