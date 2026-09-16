"""CPU-only helpers for documenting Transformer Emotion Analysis with Gaze.

Import from a checkout with ``PYTHONPATH=examples`` (the Makefile sets this).
Nothing here downloads BERT/RoBERTa or reads MATLAB files.
"""

from .alignment import (
    SUBJECT_3_SKIP_ORIGINAL,
    compacted_to_original_subject3,
    first_sentlen_mismatch,
    original_to_compacted_subject3,
    subject3_alignment_report,
)
from .fusion import FusionConfig, NumpyEyeTrackingFusion, softmax
from .gaze import collinear_pairs, feature_label_correlations, pearson_matrix
from .io import (
    load_full_sst_combined,
    load_full_sst_split,
    load_provo,
    load_subject_sentence_gaze,
    load_word_averages,
    load_zuco_combined,
    load_zuco_split,
    load_zuco_text,
)
from .metrics import calculate_metrics
from .paths import repo_root
from .pipeline import join_zuco_text_and_gaze, reconstruct_zuco_combined
from .schema import (
    FULL_SST_GAZE_COLS,
    LABEL_NAMES,
    SENTIMENT_FROM_STRING,
    ZUCO_MODEL_GAZE_COLS,
)
from .splits import disjoint_id_sets, split_inventory
from .stats import frame_profile, label_counts

__all__ = [
    "FULL_SST_GAZE_COLS",
    "FusionConfig",
    "LABEL_NAMES",
    "NumpyEyeTrackingFusion",
    "SENTIMENT_FROM_STRING",
    "SUBJECT_3_SKIP_ORIGINAL",
    "ZUCO_MODEL_GAZE_COLS",
    "calculate_metrics",
    "collinear_pairs",
    "compacted_to_original_subject3",
    "disjoint_id_sets",
    "feature_label_correlations",
    "first_sentlen_mismatch",
    "frame_profile",
    "join_zuco_text_and_gaze",
    "label_counts",
    "load_full_sst_combined",
    "load_full_sst_split",
    "load_provo",
    "load_subject_sentence_gaze",
    "load_word_averages",
    "load_zuco_combined",
    "load_zuco_split",
    "load_zuco_text",
    "original_to_compacted_subject3",
    "pearson_matrix",
    "reconstruct_zuco_combined",
    "repo_root",
    "softmax",
    "split_inventory",
    "subject3_alignment_report",
]
