"""Helpers for the personal ZuCo + SST gaze-and-transformer experiments.

This package is documentation-first. It extracts the feature names, label
maps, scaling rules, and late-fusion idea used by the original training
scripts so examples and tests can run without downloading BERT or RoBERTa.
"""

from .constants import (
    FUSION_GAZE_FEATURES,
    LABEL_ID_TO_NAME,
    LABEL_NAME_TO_ID,
    SENTENCE_ET_FEATURES,
    WORD_ET_FEATURES,
    label_name,
)
from .datasets import (
    DatasetSummary,
    feature_matrix,
    load_csv_rows,
    summarize_rows,
)
from .fusion import GazeFusionClassifier
from .metrics import classification_report, weighted_scores
from .scaling import fill_missing, mean_normalize, min_max_scale, standard_scale

__all__ = [
    "FUSION_GAZE_FEATURES",
    "LABEL_ID_TO_NAME",
    "LABEL_NAME_TO_ID",
    "SENTENCE_ET_FEATURES",
    "WORD_ET_FEATURES",
    "DatasetSummary",
    "GazeFusionClassifier",
    "classification_report",
    "feature_matrix",
    "fill_missing",
    "label_name",
    "load_csv_rows",
    "mean_normalize",
    "min_max_scale",
    "standard_scale",
    "summarize_rows",
    "weighted_scores",
]

__version__ = "0.1.0"
