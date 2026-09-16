"""Personal helpers for exploring this transformer + gaze sentiment repo.

The original training scripts stay as they were. This package exists so docs and
examples can talk about the same column names, splits, and metrics without
copy-pasting one-off notebook cells.
"""

from .features import (
    FeatureReport,
    compare_scaling,
    correlation_with_label,
    feature_correlation_matrix,
    scale_numeric_frame,
    summarize_numeric,
    word_to_sentence_means,
)
from .io import (
    DatasetBundle,
    load_csv,
    load_full_sst_splits,
    load_gaze_prediction_sample,
    load_subject_sentence_et,
    load_word_averages,
    load_zuco_combined,
    load_zuco_sentiment,
    load_zuco_splits,
    repo_root,
)
from .metrics import classification_metrics, majority_baseline
from .paths import DataPaths
from .schema import (
    FEATURE_GLOSSARY,
    SENTIMENT_ID_TO_NAME,
    SENTIMENT_NAME_TO_ID,
    SST_MODEL_FEATURES,
    ZUCO_MODEL_FEATURES,
    ZUCO_SENTENCE_ET_COLUMNS,
    label_name,
)

__all__ = [
    "DataPaths",
    "DatasetBundle",
    "FEATURE_GLOSSARY",
    "FeatureReport",
    "SENTIMENT_ID_TO_NAME",
    "SENTIMENT_NAME_TO_ID",
    "SST_MODEL_FEATURES",
    "ZUCO_MODEL_FEATURES",
    "ZUCO_SENTENCE_ET_COLUMNS",
    "classification_metrics",
    "compare_scaling",
    "correlation_with_label",
    "feature_correlation_matrix",
    "label_name",
    "load_csv",
    "load_full_sst_splits",
    "load_gaze_prediction_sample",
    "load_subject_sentence_et",
    "load_word_averages",
    "load_zuco_combined",
    "load_zuco_sentiment",
    "load_zuco_splits",
    "majority_baseline",
    "repo_root",
    "scale_numeric_frame",
    "summarize_numeric",
    "word_to_sentence_means",
]

__version__ = "0.1.0"
