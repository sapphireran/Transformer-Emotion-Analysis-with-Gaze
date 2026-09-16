"""Personal helpers for Transformer Emotion Analysis with Gaze.

This package documents and reuses the datasets already checked into this
repository. It does not download ZuCo MATLAB files or Hugging Face weights.
The original training scripts (`model_ZuCo_SST.py`, `model_full_SST.py`) stay
the source of truth for the BERT/RoBERTa fusion experiments.
"""

from tea_gaze.features import (
    CORE_FUSION_FEATURES,
    SENTIMENT_LABELS,
    FeatureSpec,
    canonicalize_gaze_columns,
    fusion_feature_frame,
    get_feature,
    list_features,
)
from tea_gaze.paths import DATASETS, repo_root

__version__ = "0.1.0"
__all__ = [
    "CORE_FUSION_FEATURES",
    "DATASETS",
    "SENTIMENT_LABELS",
    "FeatureSpec",
    "canonicalize_gaze_columns",
    "fusion_feature_frame",
    "get_feature",
    "list_features",
    "repo_root",
]
