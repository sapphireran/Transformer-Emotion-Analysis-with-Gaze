"""Absolute paths to the committed data directories.

Every example script imports from here so they work regardless of cwd,
as long as this file stays at ``<repo>/examples/paths.py``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SST_DIR = REPO_ROOT / "SST_data"
ZUCO_SST_DIR = REPO_ROOT / "ZuCo_SST_data"
ZUCO_ET_DIR = REPO_ROOT / "ZuCo_et_csv_data"
ZUCO_WORD_DIR = ZUCO_ET_DIR / "word"
GAZE_PRED_DIR = REPO_ROOT / "gaze_prediction" / "data"
RESULT_DIR = REPO_ROOT / "result"
EXAMPLES_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXAMPLES_DIR / "output"

# Files the two training scripts actually open.
FULL_SST_TRAIN = SST_DIR / "train_full_sst.csv"
FULL_SST_VALID = SST_DIR / "valid_full_sst.csv"
FULL_SST_TEST = SST_DIR / "test_full_sst.csv"
FULL_SST_COMBINED = SST_DIR / "combined_full_sst_et.csv"
FULL_SST_RAW = SST_DIR / "stts_all_sentence_level.csv"
FULL_SST_WORD_TEMPLATE = SST_DIR / "sst_et_test.csv"

ZUCO_TEXT = ZUCO_SST_DIR / "ssts_ZuCo.csv"
ZUCO_STANDARD = ZUCO_SST_DIR / "combined_sst_et_standard.csv"
ZUCO_MINMAX = ZUCO_SST_DIR / "combined_sst_et_min_max.csv"
ZUCO_TRAIN = ZUCO_SST_DIR / "train.csv"
ZUCO_VALID = ZUCO_SST_DIR / "valid.csv"
ZUCO_TEST = ZUCO_SST_DIR / "test.csv"

ZUCO_ET_AVERAGE = ZUCO_ET_DIR / "average_data.csv"
ZUCO_ET_MINMAX = ZUCO_ET_DIR / "min_max_scaled_average_data.csv"
ZUCO_ET_STANDARD = ZUCO_ET_DIR / "standard_scaled_average_data.csv"

WORD_AVERAGES_V2 = ZUCO_WORD_DIR / "word_averages_v2.csv"
PRED_TEST = GAZE_PRED_DIR / "prediction_test.csv"
PRED_TEST_V2 = GAZE_PRED_DIR / "prediction_test_v2.csv"
PROVO = GAZE_PRED_DIR / "provo.csv"

SENTIMENT_NAME = {0: "negative", 1: "neutral", 2: "positive"}

# Five-feature orders as consumed by the two training scripts.
ZUCO_ET_COLS = ["nFixations", "FFD", "GPT", "TRT", "GD"]
SST_ET_COLS = ["nFix", "FFD", "GPT", "TRT", "GD"]


def subject_sentence_csv(subject_one_based: int) -> Path:
    return ZUCO_ET_DIR / f"{subject_one_based}_SR.csv"


def subject_word_csv(subject_one_based: int) -> Path:
    return ZUCO_WORD_DIR / f"{subject_one_based}_SR.csv"
