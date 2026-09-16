"""Resolved locations for the checked-in CSVs.

Producer scripts in the repo root still mention folders that are not in this
clone (``ZuCo_mat_data/``, ``et_csv_data/``, ``ZuCo_SST_data/all/``). The
paths below are the consumer tables that *are* present.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _p(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


ZUCO_SST_DIR = _p("ZuCo_SST_data")
ZUCO_ET_DIR = _p("ZuCo_et_csv_data")
ZUCO_WORD_DIR = ZUCO_ET_DIR / "word"
SST_DIR = _p("SST_data")
GAZE_PRED_DIR = _p("gaze_prediction") / "data"
RESULT_DIR = _p("result")

ZUCO_COMBINED_STANDARD = ZUCO_SST_DIR / "combined_sst_et_standard.csv"
ZUCO_COMBINED_MINMAX = ZUCO_SST_DIR / "combined_sst_et_min_max.csv"
ZUCO_SENTENCES = ZUCO_SST_DIR / "ssts_ZuCo.csv"
ZUCO_TRAIN = ZUCO_SST_DIR / "train.csv"
ZUCO_VALID = ZUCO_SST_DIR / "valid.csv"
ZUCO_TEST = ZUCO_SST_DIR / "test.csv"

SST_COMBINED = SST_DIR / "combined_full_sst_et.csv"
SST_TRAIN = SST_DIR / "train_full_sst.csv"
SST_VALID = SST_DIR / "valid_full_sst.csv"
SST_TEST = SST_DIR / "test_full_sst.csv"
SST_RAW_SENTENCES = SST_DIR / "stts_all_sentence_level.csv"
SST_WORD_PLACEHOLDER = SST_DIR / "sst_et_test.csv"

ET_AVERAGE = ZUCO_ET_DIR / "average_data.csv"
ET_AVERAGE_STANDARD = ZUCO_ET_DIR / "standard_scaled_average_data.csv"
ET_AVERAGE_MINMAX = ZUCO_ET_DIR / "min_max_scaled_average_data.csv"
WORD_AVERAGES = ZUCO_WORD_DIR / "word_averages_v2.csv"

PROVO = GAZE_PRED_DIR / "provo.csv"
PRED_TEST = GAZE_PRED_DIR / "prediction_test.csv"
PRED_TEST_V2 = GAZE_PRED_DIR / "prediction_test_v2.csv"

MODEL_ZUCO = ROOT / "model_ZuCo_SST.py"
MODEL_FULL_SST = ROOT / "model_full_SST.py"


def subject_sentence_csv(subject_one_indexed: int) -> Path:
    if subject_one_indexed < 1 or subject_one_indexed > 12:
        raise ValueError("subject files are numbered 1..12")
    return ZUCO_ET_DIR / f"{subject_one_indexed}_SR.csv"


def subject_word_csv(subject_one_indexed: int) -> Path:
    if subject_one_indexed < 1 or subject_one_indexed > 12:
        raise ValueError("subject files are numbered 1..12")
    return ZUCO_WORD_DIR / f"{subject_one_indexed}_SR.csv"


DATASETS = {
    "zuco_combined_standard": ZUCO_COMBINED_STANDARD,
    "zuco_combined_minmax": ZUCO_COMBINED_MINMAX,
    "zuco_sentences": ZUCO_SENTENCES,
    "zuco_train": ZUCO_TRAIN,
    "zuco_valid": ZUCO_VALID,
    "zuco_test": ZUCO_TEST,
    "sst_combined": SST_COMBINED,
    "sst_train": SST_TRAIN,
    "sst_valid": SST_VALID,
    "sst_test": SST_TEST,
    "et_average": ET_AVERAGE,
    "et_average_standard": ET_AVERAGE_STANDARD,
    "word_averages": WORD_AVERAGES,
    "provo": PROVO,
    "pred_test": PRED_TEST,
}
