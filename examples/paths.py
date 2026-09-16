"""Repository paths used by every example script."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SST_DIR = REPO_ROOT / "SST_data"
ZUCO_SST_DIR = REPO_ROOT / "ZuCo_SST_data"
ZUCO_ET_DIR = REPO_ROOT / "ZuCo_et_csv_data"
ZUCO_WORD_DIR = ZUCO_ET_DIR / "word"
GAZE_PRED_DIR = REPO_ROOT / "gaze_prediction" / "data"

SST_COMBINED = SST_DIR / "combined_full_sst_et.csv"
SST_TRAIN = SST_DIR / "train_full_sst.csv"
SST_VALID = SST_DIR / "valid_full_sst.csv"
SST_TEST = SST_DIR / "test_full_sst.csv"
SST_WORD_ZEROS = SST_DIR / "sst_et_test.csv"
SST_HEADERLESS = SST_DIR / "stts_all_sentence_level.csv"

ZUCO_COMBINED_STD = ZUCO_SST_DIR / "combined_sst_et_standard.csv"
ZUCO_COMBINED_MM = ZUCO_SST_DIR / "combined_sst_et_min_max.csv"
ZUCO_LABELS = ZUCO_SST_DIR / "ssts_ZuCo.csv"
ZUCO_TRAIN = ZUCO_SST_DIR / "train.csv"
ZUCO_VALID = ZUCO_SST_DIR / "valid.csv"
ZUCO_TEST = ZUCO_SST_DIR / "test.csv"

ZUCO_AVERAGE = ZUCO_ET_DIR / "average_data.csv"
ZUCO_AVERAGE_STD = ZUCO_ET_DIR / "standard_scaled_average_data.csv"
ZUCO_AVERAGE_MM = ZUCO_ET_DIR / "min_max_scaled_average_data.csv"
ZUCO_SUBJECT_1 = ZUCO_ET_DIR / "1_SR.csv"
ZUCO_SUBJECT_3 = ZUCO_ET_DIR / "3_SR.csv"
ZUCO_WORD_AVG = ZUCO_WORD_DIR / "word_averages_v2.csv"
ZUCO_WORD_SUBJECT_1 = ZUCO_WORD_DIR / "1_SR.csv"

PRED_V2 = GAZE_PRED_DIR / "prediction_test_v2.csv"
PRED_SMALL = GAZE_PRED_DIR / "prediction_test.csv"
PROVO = GAZE_PRED_DIR / "provo.csv"

SST_GAZE_COLS = ("nFix", "GD", "TRT", "FFD", "GPT")
ZUCO_MODEL_GAZE_COLS = ("nFixations", "FFD", "GPT", "TRT", "GD")
ZUCO_ALL_GAZE_COLS = (
    "omissionRate",
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
)
AVERAGE_NUMERIC_COLS = (
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

LABEL_NAMES = {"0": "negative", "1": "neutral", "2": "positive"}
