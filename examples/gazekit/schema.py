"""Column names, label maps, and rename rules for every checked-in table."""

from __future__ import annotations

from typing import Mapping

# Integer convention used by every fused modeling CSV.
SENTIMENT_TO_INT: Mapping[str, int] = {
    "NEGATIVE": 0,
    "NEUTRAL": 1,
    "POSITIVE": 2,
}

SENTIMENT_FROM_INT: Mapping[int, str] = {v: k for k, v in SENTIMENT_TO_INT.items()}

LABEL_NAMES = ("NEGATIVE", "NEUTRAL", "POSITIVE")

# The five channels EyeTrackingModel actually consumes, in constructor order.
CANONICAL_GAZE = ("nFixations", "FFD", "GPT", "TRT", "GD")

# Present on ZuCo sentence tables but not fed to the transformer head.
EXTRA_SENTENCE_ET = ("omissionRate", "meanPupilSize", "SFD", "SentLen")

# Full-SST fused tables use a shorter fixation-count name.
FULL_SST_GAZE = ("nFix", "FFD", "GPT", "TRT", "GD")

ZUCO_TO_CANONICAL = {
    "nFixations": "nFixations",
    "nFix": "nFixations",
    "FFD": "FFD",
    "GPT": "GPT",
    "TRT": "TRT",
    "GD": "GD",
}

WORD_GAZE = ("nFixations", "meanPupilSize", "GD", "TRT", "FFD", "SFD", "GPT")

# DataTransformer sentence fields, in allocation order.
TRANSFORMER_SENTENCE_FIELDS = (
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

# Subject-level exclusions copied from utils_ZuCo.DataTransformer.__call__.
# Keys are (task, 0-based subject). Values are human-readable drop rules.
KNOWN_BAD_TRIALS = {
    ("task1", 2): "drop i in [150, 249] and i == 399",
    ("task2", 6): "drop i <= 49",
    ("task2", 11): "drop i in [50, 99]",
    ("task3", 3): "drop i in [178, 224]",
    ("task3", 7): "drop i >= 359",
    ("task3", 11): "drop i in [270, 313] and i in [362, 406]",
}

# Expected row counts on this checkout (header excluded). Used as smoke checks.
EXPECTED_ROWS = {
    "zuco_sst_combined": 400,
    "zuco_sst_train": 320,
    "zuco_sst_valid": 40,
    "zuco_sst_test": 40,
    "zuco_subject_default": 400,
    "zuco_subject_3": 299,
    "full_sst_train": 9482,
    "full_sst_valid": 1185,
    "full_sst_test": 1186,
}


def gaze_columns_in(frame_columns) -> list[str]:
    """Return canonical gaze names that are present under any known alias."""
    present = set(frame_columns)
    found: list[str] = []
    for canonical in CANONICAL_GAZE:
        aliases = [src for src, dst in ZUCO_TO_CANONICAL.items() if dst == canonical]
        if any(alias in present for alias in aliases):
            found.append(canonical)
    return found
