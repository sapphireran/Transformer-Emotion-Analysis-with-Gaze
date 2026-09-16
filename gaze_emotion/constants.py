"""Shared names for labels and eye-tracking features.

These names match the CSV headers produced by the ZuCo conversion scripts
and consumed by ``model_ZuCo_SST.py`` / ``model_full_SST.py``.
"""

from __future__ import annotations

from typing import Mapping

# Stanford Sentiment Treebank style 3-way labels used throughout this repo.
LABEL_NAME_TO_ID: Mapping[str, int] = {
    "NEGATIVE": 0,
    "NEUTRAL": 1,
    "POSITIVE": 2,
}

LABEL_ID_TO_NAME: Mapping[int, str] = {v: k for k, v in LABEL_NAME_TO_ID.items()}

# Sentence-level columns written by ``utils_ZuCo.DataTransformer`` (task1, SR).
SENTENCE_ET_FEATURES: tuple[str, ...] = (
    "SentLen",
    "omissionRate",
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
)

# Word-level columns written by the same transformer.
WORD_ET_FEATURES: tuple[str, ...] = (
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
    "WordLen",
)

# The five gaze channels the fusion models actually concatenate onto BERT/RoBERTa.
# Full-SST CSVs rename nFixations -> nFix; ZuCo CSVs keep nFixations.
FUSION_GAZE_FEATURES: tuple[str, ...] = ("nFixations", "FFD", "GPT", "TRT", "GD")
FUSION_GAZE_FEATURES_FULL_SST: tuple[str, ...] = ("nFix", "FFD", "GPT", "TRT", "GD")

# Default hyperparameters copied from the original training scripts so examples
# can discuss them without opening those files.
DEFAULT_HIDDEN_LAYER_SIZE = 16
DEFAULT_NUM_LABELS = 3
DEFAULT_NUM_EYE_TRACKING_FEATURES = 5
DEFAULT_MAX_LENGTH = 128
DEFAULT_DROPOUT = 0.1
DEFAULT_LEARNING_RATE = 5e-5

ZUCO_SUBJECT_COUNT = 12
ZUCO_TASK1_EXPECTED_SENTENCES = 400


def label_name(label: int | str) -> str:
    """Return a human-readable label for an id or already-named value."""
    if isinstance(label, str):
        key = label.strip().upper()
        if key in LABEL_NAME_TO_ID:
            return key
        if key.lstrip("-").isdigit():
            return label_name(int(key))
        raise KeyError(f"Unknown label name: {label!r}")
    try:
        return LABEL_ID_TO_NAME[int(label)]
    except (KeyError, ValueError) as exc:
        raise KeyError(f"Unknown label id: {label!r}") from exc


def fusion_feature_names(style: str = "zuco") -> tuple[str, ...]:
    """Return the five fusion channels for a dataset family."""
    if style == "zuco":
        return FUSION_GAZE_FEATURES
    if style == "full_sst":
        return FUSION_GAZE_FEATURES_FULL_SST
    raise ValueError("style must be 'zuco' or 'full_sst'")
