"""Shared helpers for the personal example suite."""

from .paths import REPO_ROOT, DatasetSpec, iter_dataset_specs
from .loaders import (
    FUSION_GAZE_COLUMNS,
    LABEL_NAMES,
    load_csv,
    load_headerless_sst,
    load_zuco_experiment,
    word_rows_for_sentence,
)
from .metrics import MetricBundle, majority_baseline, score_predictions
from .fusion import hashed_text_matrix, run_fusion_cv

__all__ = [
    "REPO_ROOT",
    "DatasetSpec",
    "iter_dataset_specs",
    "FUSION_GAZE_COLUMNS",
    "LABEL_NAMES",
    "load_csv",
    "load_headerless_sst",
    "load_zuco_experiment",
    "word_rows_for_sentence",
    "MetricBundle",
    "majority_baseline",
    "score_predictions",
    "hashed_text_matrix",
    "run_fusion_cv",
]
