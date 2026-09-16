"""Shared paths and column lists for the personal example scripts.

Training files execute I/O at import time, so examples do not import them.
Keep fusion-feature order identical to model_ZuCo_SST.py / model_full_SST.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

ZUCO_COMBINED_STANDARD = REPO_ROOT / "ZuCo_SST_data" / "combined_sst_et_standard.csv"
ZUCO_COMBINED_MINMAX = REPO_ROOT / "ZuCo_SST_data" / "combined_sst_et_min_max.csv"
ZUCO_SENTENCES = REPO_ROOT / "ZuCo_SST_data" / "ssts_ZuCo.csv"
ZUCO_TRAIN = REPO_ROOT / "ZuCo_SST_data" / "train.csv"
ZUCO_VALID = REPO_ROOT / "ZuCo_SST_data" / "valid.csv"
ZUCO_TEST = REPO_ROOT / "ZuCo_SST_data" / "test.csv"

ZUCO_ET_DIR = REPO_ROOT / "ZuCo_et_csv_data"
ZUCO_AVERAGE_RAW = ZUCO_ET_DIR / "average_data.csv"
ZUCO_AVERAGE_STANDARD = ZUCO_ET_DIR / "standard_scaled_average_data.csv"
ZUCO_AVERAGE_MINMAX = ZUCO_ET_DIR / "min_max_scaled_average_data.csv"
ZUCO_WORD_AVERAGE = ZUCO_ET_DIR / "word" / "word_averages_v2.csv"

SST_COMBINED = REPO_ROOT / "SST_data" / "combined_full_sst_et.csv"
SST_TRAIN = REPO_ROOT / "SST_data" / "train_full_sst.csv"
SST_VALID = REPO_ROOT / "SST_data" / "valid_full_sst.csv"
SST_TEST = REPO_ROOT / "SST_data" / "test_full_sst.csv"
SST_RAW_LABELS = REPO_ROOT / "SST_data" / "stts_all_sentence_level.csv"
SST_WORD_PLACEHOLDER = REPO_ROOT / "SST_data" / "sst_et_test.csv"

PRED_TEST_V2 = REPO_ROOT / "gaze_prediction" / "data" / "prediction_test_v2.csv"
PRED_TEST = REPO_ROOT / "gaze_prediction" / "data" / "prediction_test.csv"
PROVO = REPO_ROOT / "gaze_prediction" / "data" / "provo.csv"

# Order consumed by EyeTrackingModel in both training scripts.
ZUCO_FUSION_COLUMNS = ["nFixations", "FFD", "GPT", "TRT", "GD"]
SST_FUSION_COLUMNS = ["nFix", "FFD", "GPT", "TRT", "GD"]

ZUCO_EXTRA_GAZE = ["omissionRate", "meanPupilSize", "SFD"]
SST_ON_DISK_GAZE_ORDER = ["nFix", "GD", "TRT", "FFD", "GPT"]

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

INVENTORY = {
    "ZuCo_SST_data/combined_sst_et_standard.csv": ZUCO_COMBINED_STANDARD,
    "ZuCo_SST_data/combined_sst_et_min_max.csv": ZUCO_COMBINED_MINMAX,
    "ZuCo_SST_data/ssts_ZuCo.csv": ZUCO_SENTENCES,
    "ZuCo_SST_data/train.csv": ZUCO_TRAIN,
    "ZuCo_SST_data/valid.csv": ZUCO_VALID,
    "ZuCo_SST_data/test.csv": ZUCO_TEST,
    "ZuCo_et_csv_data/average_data.csv": ZUCO_AVERAGE_RAW,
    "ZuCo_et_csv_data/standard_scaled_average_data.csv": ZUCO_AVERAGE_STANDARD,
    "ZuCo_et_csv_data/min_max_scaled_average_data.csv": ZUCO_AVERAGE_MINMAX,
    "ZuCo_et_csv_data/word/word_averages_v2.csv": ZUCO_WORD_AVERAGE,
    "SST_data/combined_full_sst_et.csv": SST_COMBINED,
    "SST_data/train_full_sst.csv": SST_TRAIN,
    "SST_data/valid_full_sst.csv": SST_VALID,
    "SST_data/test_full_sst.csv": SST_TEST,
    "SST_data/stts_all_sentence_level.csv": SST_RAW_LABELS,
    "SST_data/sst_et_test.csv": SST_WORD_PLACEHOLDER,
    "gaze_prediction/data/prediction_test_v2.csv": PRED_TEST_V2,
    "gaze_prediction/data/prediction_test.csv": PRED_TEST,
    "gaze_prediction/data/provo.csv": PROVO,
}

EXPECTED_COLUMNS = {
    "ZuCo_SST_data/combined_sst_et_standard.csv": [
        "sentence_id",
        "sentence",
        "sentiment_label",
        "omissionRate",
        "nFixations",
        "meanPupilSize",
        "GD",
        "TRT",
        "FFD",
        "SFD",
        "GPT",
    ],
    "ZuCo_SST_data/combined_sst_et_min_max.csv": [
        "sentence_id",
        "sentence",
        "sentiment_label",
        "omissionRate",
        "nFixations",
        "meanPupilSize",
        "GD",
        "TRT",
        "FFD",
        "SFD",
        "GPT",
    ],
    "ZuCo_SST_data/ssts_ZuCo.csv": ["sentence_id", "sentence", "sentiment_label"],
    "SST_data/combined_full_sst_et.csv": [
        "sentence_id",
        "sentence",
        "sentiment_label",
        "nFix",
        "GD",
        "TRT",
        "FFD",
        "GPT",
    ],
    "SST_data/train_full_sst.csv": [
        "sentence_id",
        "sentence",
        "sentiment_label",
        "nFix",
        "GD",
        "TRT",
        "FFD",
        "GPT",
    ],
    "SST_data/valid_full_sst.csv": [
        "sentence_id",
        "sentence",
        "sentiment_label",
        "nFix",
        "GD",
        "TRT",
        "FFD",
        "GPT",
    ],
    "SST_data/test_full_sst.csv": [
        "sentence_id",
        "sentence",
        "sentiment_label",
        "nFix",
        "GD",
        "TRT",
        "FFD",
        "GPT",
    ],
    "ZuCo_et_csv_data/average_data.csv": [
        "id",
        "SentLen",
        "omissionRate",
        "nFixations",
        "meanPupilSize",
        "GD",
        "TRT",
        "FFD",
        "SFD",
        "GPT",
    ],
    "gaze_prediction/data/prediction_test_v2.csv": [
        "sentence_id",
        "word_id",
        "word",
        "nFix",
        "FFD",
        "GPT",
        "TRT",
        "GD",
    ],
    "gaze_prediction/data/provo.csv": [
        "sentence_id",
        "word_id",
        "word",
        "nFix",
        "FFD",
        "GPT",
        "TRT",
        "fixProp",
    ],
}

EXPECTED_ROW_COUNTS = {
    "ZuCo_SST_data/combined_sst_et_standard.csv": 400,
    "ZuCo_SST_data/combined_sst_et_min_max.csv": 400,
    "ZuCo_SST_data/ssts_ZuCo.csv": 400,
    "ZuCo_SST_data/train.csv": 320,
    "ZuCo_SST_data/valid.csv": 40,
    "ZuCo_SST_data/test.csv": 40,
    "ZuCo_et_csv_data/average_data.csv": 400,
    "SST_data/combined_full_sst_et.csv": 11853,
    "SST_data/train_full_sst.csv": 9482,
    "SST_data/valid_full_sst.csv": 1185,
    "SST_data/test_full_sst.csv": 1186,
    "SST_data/sst_et_test.csv": 191971,
    "gaze_prediction/data/prediction_test_v2.csv": 191971,
    "gaze_prediction/data/prediction_test.csv": 1751,
    "gaze_prediction/data/provo.csv": 2659,
    "ZuCo_et_csv_data/word/word_averages_v2.csv": 7129,
}


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.relative_to(REPO_ROOT)}")
    return pd.read_csv(path, **kwargs)


def subject_sentence_paths() -> list[Path]:
    return [ZUCO_ET_DIR / f"{i}_SR.csv" for i in range(1, 13)]


def subject_word_paths() -> list[Path]:
    return [ZUCO_ET_DIR / "word" / f"{i}_SR.csv" for i in range(1, 13)]


def label_name_series(labels: pd.Series) -> pd.Series:
    return labels.map(LABEL_NAMES).fillna("unknown")


def write_text(path: Path, text: str) -> Path:
    ensure_output_dir()
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path
