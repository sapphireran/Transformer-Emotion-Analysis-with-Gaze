"""Helpers for the personal analysis examples.

These modules only depend on pandas / numpy / scikit-learn. They never
import the original training scripts (those pull in transformers).
"""

from .baselines import holdout_scores, permute_anova, stratified_cv_scores
from .fusion import FusionWalkthrough
from .loading import (
    LABEL_NAMES,
    SUBJECT_SENTENCE_ROWS,
    DatasetSpec,
    documented_datasets,
    load_dataset,
    load_full_sst_splits,
    load_sst_raw,
    load_subject_sentence_tables,
    load_zuco_combined,
    load_zuco_word_average,
    remap_subject3_original_ids,
)
from .paths import repo_root, resolve_root

__all__ = [
    "LABEL_NAMES",
    "DatasetSpec",
    "FusionWalkthrough",
    "SUBJECT_SENTENCE_ROWS",
    "documented_datasets",
    "holdout_scores",
    "load_dataset",
    "load_full_sst_splits",
    "load_sst_raw",
    "load_subject_sentence_tables",
    "load_zuco_combined",
    "load_zuco_word_average",
    "permute_anova",
    "remap_subject3_original_ids",
    "repo_root",
    "resolve_root",
    "stratified_cv_scores",
]
