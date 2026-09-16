"""Named gaze feature sets and class-conditional summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from . import schema, stats
from .io_csv import Row, float_column, select_columns, to_int


FEATURE_SETS = {
    "sst5": schema.SST5,
    "zuco5": schema.ZUCO5,
    "zuco_full": schema.ZUCO_FULL,
}


def labels_of(rows: Sequence[Row]) -> List[int]:
    return [to_int(row["sentiment_label"]) for row in rows]


def matrix_for(rows: Sequence[Row], feature_set: str) -> List[List[float]]:
    return select_columns(rows, FEATURE_SETS[feature_set])


def class_means(
    rows: Sequence[Row], feature_set: str
) -> Dict[int, Dict[str, float]]:
    names = FEATURE_SETS[feature_set]
    buckets: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        label = to_int(row["sentiment_label"])
        for name in names:
            buckets[label][name].append(float(row[name]))
    return {
        label: {name: stats.mean(values) for name, values in named.items()}
        for label, named in sorted(buckets.items())
    }


def association_scores(
    rows: Sequence[Row], feature_set: str
) -> List[Tuple[str, float]]:
    """Rank columns by mean absolute difference of class-conditional means.

    This is a descriptive ranking, not a hypothesis test.
    """
    names = FEATURE_SETS[feature_set]
    means = class_means(rows, feature_set)
    labels = sorted(means)
    scores: List[Tuple[str, float]] = []
    for name in names:
        values = [means[label][name] for label in labels]
        if len(values) < 2:
            scores.append((name, 0.0))
            continue
        pair_gaps = [
            abs(values[i] - values[j])
            for i in range(len(values))
            for j in range(i + 1, len(values))
        ]
        scores.append((name, sum(pair_gaps) / len(pair_gaps)))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


def label_correlations(
    rows: Sequence[Row], feature_set: str
) -> List[Tuple[str, float]]:
    y = [float(v) for v in labels_of(rows)]
    scored = []
    for name in FEATURE_SETS[feature_set]:
        x = float_column(rows, name)
        scored.append((name, stats.pearson(x, y)))
    scored.sort(key=lambda item: abs(item[1]), reverse=True)
    return scored
