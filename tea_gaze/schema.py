"""Column names, label maps, and a glossary for the personal datasets here.

The original training scripts do not share a single schema. Full SST uses
``nFix`` while ZuCo sentence files use ``nFixations``. Keep both spellings
explicit so examples do not silently rename columns.
"""

from __future__ import annotations

from typing import Mapping

SENTIMENT_ID_TO_NAME: dict[int, str] = {
    0: "NEGATIVE",
    1: "NEUTRAL",
    2: "POSITIVE",
}

SENTIMENT_NAME_TO_ID: dict[str, int] = {
    name: idx for idx, name in SENTIMENT_ID_TO_NAME.items()
}

# Five features concatenated onto BERT/RoBERTa pooler output in the training scripts.
ZUCO_MODEL_FEATURES: tuple[str, ...] = (
    "nFixations",
    "FFD",
    "GPT",
    "TRT",
    "GD",
)

SST_MODEL_FEATURES: tuple[str, ...] = (
    "nFix",
    "FFD",
    "GPT",
    "TRT",
    "GD",
)

ZUCO_SENTENCE_ET_COLUMNS: tuple[str, ...] = (
    "omissionRate",
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
)

ZUCO_WORD_ET_COLUMNS: tuple[str, ...] = (
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
)

WORD_IDENTITY_COLUMNS: tuple[str, ...] = (
    "id",
    "Sent_ID",
    "Word_ID",
    "Word",
    "WordLen",
)

FEATURE_GLOSSARY: dict[str, str] = {
    "nFix": "Number of fixations on the unit (word or sentence mean).",
    "nFixations": "ZuCo spelling of nFix: fixation count, usually averaged over fixated words.",
    "FFD": "First Fixation Duration: time of the first fixation on the unit.",
    "GD": "Gaze Duration / first-pass time: sum of first-pass fixations before leaving the unit.",
    "TRT": "Total Reading Time: all fixations on the unit, including regressions back in.",
    "GPT": "Go-Past Time / regression-path duration: time from first entering a unit until it is left to the right.",
    "SFD": "Single Fixation Duration: duration when the unit received exactly one fixation.",
    "meanPupilSize": "Mean pupil size reported by ZuCo for the unit.",
    "omissionRate": "Fraction of words in the sentence that were not fixated.",
    "SentLen": "Sentence length in words as stored in the ZuCo sentence export.",
    "WordLen": "Character length of the token after light punctuation stripping.",
    "sentiment_label": "3-class sentiment: 0 negative, 1 neutral, 2 positive.",
    "sentence_id": "Stable sentence index used to join text and eye-tracking tables.",
    "fixProp": "PROVO-style fixation proportion; not used by the transformer trainers.",
}

MODEL_TYPES: tuple[str, ...] = (
    "bert",
    "roberta",
    "bert_eye_tracking",
    "roberta_eye_tracking",
)

HIDDEN_LAYER_SIZE = 16
TRANSFORMER_HIDDEN_SIZE = 768
NUM_LABELS = 3
NUM_EYE_TRACKING_FEATURES = 5


def label_name(label: int) -> str:
    """Return NEGATIVE / NEUTRAL / POSITIVE for a numeric class id."""
    try:
        return SENTIMENT_ID_TO_NAME[int(label)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unknown sentiment label: {label!r}") from exc


def required_columns(kind: str) -> tuple[str, ...]:
    """Column checklist used by loaders and examples."""
    mapping: Mapping[str, tuple[str, ...]] = {
        "zuco_text": ("sentence_id", "sentence", "sentiment_label"),
        "zuco_sentence_et": ("sentence_id", "sentence", "sentiment_label") + ZUCO_SENTENCE_ET_COLUMNS,
        "full_sst": ("sentence_id", "sentence", "sentiment_label") + SST_MODEL_FEATURES,
        "zuco_word": WORD_IDENTITY_COLUMNS[:4] + ZUCO_WORD_ET_COLUMNS,
    }
    if kind not in mapping:
        raise KeyError(f"Unknown schema kind {kind!r}. Expected one of {sorted(mapping)}")
    return mapping[kind]
