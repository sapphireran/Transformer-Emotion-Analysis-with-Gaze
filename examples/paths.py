"""Repository-root paths for the CPU example scripts.

Every example should be launched from the repository root:

    python3 examples/inspect_datasets.py

Paths are resolved from this file so an accidental cwd change still works.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ZUCO_COMBINED_STANDARD = REPO_ROOT / "ZuCo_SST_data" / "combined_sst_et_standard.csv"
ZUCO_COMBINED_MINMAX = REPO_ROOT / "ZuCo_SST_data" / "combined_sst_et_min_max.csv"
ZUCO_TEXT = REPO_ROOT / "ZuCo_SST_data" / "ssts_ZuCo.csv"
ZUCO_TRAIN = REPO_ROOT / "ZuCo_SST_data" / "train.csv"
ZUCO_VALID = REPO_ROOT / "ZuCo_SST_data" / "valid.csv"
ZUCO_TEST = REPO_ROOT / "ZuCo_SST_data" / "test.csv"

ZUCO_AVERAGE_RAW = REPO_ROOT / "ZuCo_et_csv_data" / "average_data.csv"
ZUCO_AVERAGE_STANDARD = REPO_ROOT / "ZuCo_et_csv_data" / "standard_scaled_average_data.csv"
ZUCO_AVERAGE_MINMAX = REPO_ROOT / "ZuCo_et_csv_data" / "min_max_scaled_average_data.csv"
ZUCO_SUBJECT_DIR = REPO_ROOT / "ZuCo_et_csv_data"
ZUCO_WORD_AVERAGES = REPO_ROOT / "ZuCo_et_csv_data" / "word" / "word_averages_v2.csv"

SST_COMBINED = REPO_ROOT / "SST_data" / "combined_full_sst_et.csv"
SST_TRAIN = REPO_ROOT / "SST_data" / "train_full_sst.csv"
SST_VALID = REPO_ROOT / "SST_data" / "valid_full_sst.csv"
SST_TEST = REPO_ROOT / "SST_data" / "test_full_sst.csv"
SST_RAW_TEXT = REPO_ROOT / "SST_data" / "stts_all_sentence_level.csv"

PRED_WORD_V2 = REPO_ROOT / "gaze_prediction" / "data" / "prediction_test_v2.csv"
PROVO = REPO_ROOT / "gaze_prediction" / "data" / "provo.csv"

SAMPLE_OUTPUT_DIR = Path(__file__).resolve().parent / "sample_outputs"

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

ZUCO_FUSION_FEATURES = ["nFixations", "FFD", "GPT", "TRT", "GD"]
SST_FUSION_FEATURES = ["nFix", "FFD", "GPT", "TRT", "GD"]
ZUCO_EXTRA_FEATURES = ["omissionRate", "meanPupilSize", "SFD"]
