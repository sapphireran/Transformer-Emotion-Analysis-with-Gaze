"""Column names, label maps, and expected table sizes.

Values were measured from the committed CSVs. Tests fail if a table is
replaced with a different schema.
"""

from __future__ import annotations

from types import MappingProxyType

SENTIMENT_FROM_STRING = MappingProxyType(
    {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
)
LABEL_NAMES = MappingProxyType({0: "negative", 1: "neutral", 2: "positive"})

# Columns actually passed to EyeTrackingModel (order matters).
FULL_SST_GAZE_COLS = ("nFix", "FFD", "GPT", "TRT", "GD")
ZUCO_MODEL_GAZE_COLS = ("nFixations", "FFD", "GPT", "TRT", "GD")

ZUCO_SENTENCE_GAZE_COLS = (
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
ZUCO_JOIN_GAZE_COLS = (
    "omissionRate",
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
)
WORD_GAZE_COLS = (
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
)
PROVO_GAZE_COLS = ("nFix", "FFD", "GPT", "TRT", "fixProp")
PREDICTED_WORD_GAZE_COLS = ("nFix", "FFD", "GPT", "TRT", "GD")

ZUCO_SUBJECT_FILES = tuple(f"{i}_SR.csv" for i in range(1, 13))
ZUCO_SUBJECT_3_INDEX = 3  # 1-based filename; transformer subject index 2
ZUCO_SUBJECT_3_N_SENTENCES = 299
ZUCO_N_SENTENCES = 400
ZUCO_N_SUBJECTS = 12
FULL_SST_N_SENTENCES = 11853

ZUCO_LABEL_COUNTS = MappingProxyType({0: 123, 1: 137, 2: 140})
FULL_SST_LABEL_COUNTS = MappingProxyType({0: 4649, 1: 2241, 2: 4963})

ZUCO_SPLIT_ROWS = MappingProxyType({"train": 320, "valid": 40, "test": 40})
FULL_SST_SPLIT_ROWS = MappingProxyType(
    {"train": 9482, "valid": 1185, "test": 1186}
)

WORD_AVERAGES_ROWS = 7129
PROVO_ROWS = 2659
PREDICTION_TEST_ROWS = 1751
PREDICTION_TEST_V2_ROWS = 191971

# Task 1 subject 2 skip list from utils_ZuCo.DataTransformer.__call__
SUBJECT_3_SKIP_ORIGINAL = tuple(range(150, 250)) + (399,)
