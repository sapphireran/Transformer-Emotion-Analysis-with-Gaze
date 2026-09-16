"""Helpers for the personal analysis examples.

These modules only depend on pandas / numpy / scikit-learn. They never
import the original training scripts (those pull in transformers).
"""

from .paths import repo_root, resolve_root
from .loading import (
    LABEL_NAMES,
    DatasetSpec,
    documented_datasets,
    load_dataset,
    load_full_sst_splits,
    load_sst_raw,
    load_subject_sentence_tables,
    load_zuco_combined,
    load_zuco_word_average,
    remap_subject3_original_ids,
    SUBJECT_SENTENCE_ROWS,
)

__all__ = [
    "LABEL_NAMES",
    "DatasetSpec",
    "documented_datasets",
    "load_dataset",
    "load_full_sst_splits",
    "load_sst_raw",
    "load_subject_sentence_tables",
    "load_zuco_combined",
    "load_zuco_word_average",
    "remap_subject3_original_ids",
    "SUBJECT_SENTENCE_ROWS",
    "repo_root",
    "resolve_root",
]
