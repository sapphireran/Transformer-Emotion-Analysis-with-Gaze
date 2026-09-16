"""Repo-relative paths for the personal atlas. Keep this the single map."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SST_DIR = ROOT / "SST_data"
ZUCO_SST_DIR = ROOT / "ZuCo_SST_data"
ZUCO_ET_DIR = ROOT / "ZuCo_et_csv_data"
ZUCO_WORD_DIR = ZUCO_ET_DIR / "word"
GAZE_PRED_DIR = ROOT / "gaze_prediction" / "data"
RESULT_DIR = ROOT / "result"
OUTPUT_DIR = ROOT / "examples" / "outputs"

SST_TRAIN = SST_DIR / "train_full_sst.csv"
SST_VALID = SST_DIR / "valid_full_sst.csv"
SST_TEST = SST_DIR / "test_full_sst.csv"
SST_COMBINED = SST_DIR / "combined_full_sst_et.csv"
SST_STTS = SST_DIR / "stts_all_sentence_level.csv"
SST_ET_PLACEHOLDER = SST_DIR / "sst_et_test.csv"

ZUCO_STANDARD = ZUCO_SST_DIR / "combined_sst_et_standard.csv"
ZUCO_MINMAX = ZUCO_SST_DIR / "combined_sst_et_min_max.csv"
ZUCO_SENTENCES = ZUCO_SST_DIR / "ssts_ZuCo.csv"
ZUCO_TRAIN = ZUCO_SST_DIR / "train.csv"
ZUCO_VALID = ZUCO_SST_DIR / "valid.csv"
ZUCO_TEST = ZUCO_SST_DIR / "test.csv"

AVERAGE_SENTENCE = ZUCO_ET_DIR / "average_data.csv"
AVERAGE_MINMAX = ZUCO_ET_DIR / "min_max_scaled_average_data.csv"
AVERAGE_STANDARD = ZUCO_ET_DIR / "standard_scaled_average_data.csv"
WORD_AVERAGES = ZUCO_WORD_DIR / "word_averages.csv"
WORD_AVERAGES_V2 = ZUCO_WORD_DIR / "word_averages_v2.csv"

PREDICTION_TEST = GAZE_PRED_DIR / "prediction_test.csv"
PREDICTION_TEST_V2 = GAZE_PRED_DIR / "prediction_test_v2.csv"
PROVO = GAZE_PRED_DIR / "provo.csv"

TRAIN_PLOT = RESULT_DIR / "train_data_scatter_hist_plots.png"
TEST_PLOT = RESULT_DIR / "test_data_scatter_hist_plots.png"
PROVO_PLOT = RESULT_DIR / "provo_data_scatter_hist_plots.png"

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
WORD_GAZE_COLS = (
    "nFixations",
    "meanPupilSize",
    "GD",
    "TRT",
    "FFD",
    "SFD",
    "GPT",
)

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def subject_sentence_csv(subject_1indexed: int) -> Path:
    if not 1 <= subject_1indexed <= 12:
        raise ValueError(f"subject must be 1-12, got {subject_1indexed}")
    return ZUCO_ET_DIR / f"{subject_1indexed}_SR.csv"


def subject_word_csv(subject_1indexed: int) -> Path:
    if not 1 <= subject_1indexed <= 12:
        raise ValueError(f"subject must be 1-12, got {subject_1indexed}")
    return ZUCO_WORD_DIR / f"{subject_1indexed}_SR.csv"


def expected_tables() -> dict[str, Path]:
    """Every CSV / plot the atlas expects to exist in a full checkout."""
    tables = {
        "sst_train": SST_TRAIN,
        "sst_valid": SST_VALID,
        "sst_test": SST_TEST,
        "sst_combined": SST_COMBINED,
        "sst_stts": SST_STTS,
        "sst_et_placeholder": SST_ET_PLACEHOLDER,
        "zuco_standard": ZUCO_STANDARD,
        "zuco_minmax": ZUCO_MINMAX,
        "zuco_sentences": ZUCO_SENTENCES,
        "zuco_train": ZUCO_TRAIN,
        "zuco_valid": ZUCO_VALID,
        "zuco_test": ZUCO_TEST,
        "average_sentence": AVERAGE_SENTENCE,
        "average_minmax": AVERAGE_MINMAX,
        "average_standard": AVERAGE_STANDARD,
        "word_averages": WORD_AVERAGES,
        "word_averages_v2": WORD_AVERAGES_V2,
        "prediction_test": PREDICTION_TEST,
        "prediction_test_v2": PREDICTION_TEST_V2,
        "provo": PROVO,
        "train_plot": TRAIN_PLOT,
        "test_plot": TEST_PLOT,
        "provo_plot": PROVO_PLOT,
    }
    for i in range(1, 13):
        tables[f"subject_{i}_sentence"] = subject_sentence_csv(i)
        tables[f"subject_{i}_word"] = subject_word_csv(i)
    return tables
