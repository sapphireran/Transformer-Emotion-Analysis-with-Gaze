"""Helpers for personal docs and examples in this gaze + sentiment repo.

The training scripts stay at the repository root. This package only reads the
checked-in CSVs, computes summaries, and sketches the fusion idea without
downloading transformer weights.
"""

from .catalog import DATASETS, DatasetSpec, get_dataset, list_datasets
from .fusion import GazeFusionForward, text_vs_gaze_cv
from .io import load_dataset
from .labels import LABEL_NAMES, label_counts, label_name
from .paths import repo_root
from .profiles import WordGaze, sentence_profile, top_words_by_measure
from .scaling import scaler_report
from .splits import audit_split
from .stats import correlation_matrix, describe_numeric, grouped_means

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "GazeFusionForward",
    "LABEL_NAMES",
    "WordGaze",
    "audit_split",
    "correlation_matrix",
    "describe_numeric",
    "get_dataset",
    "grouped_means",
    "label_counts",
    "label_name",
    "list_datasets",
    "load_dataset",
    "repo_root",
    "scaler_report",
    "sentence_profile",
    "text_vs_gaze_cv",
    "top_words_by_measure",
]
