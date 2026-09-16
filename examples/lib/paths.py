"""Named locations inside this repository.

Every example script should import paths from here instead of hard-coding
relative strings. Resolution is based on this file's location so the
current working directory does not matter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXAMPLES_DIR.parent
OUTPUTS_DIR = EXAMPLES_DIR / "outputs"


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


# (relative path from repo root, short role string)
TABLES: Dict[str, Tuple[str, str]] = {
    "zuco_text": ("ZuCo_SST_data/ssts_ZuCo.csv", "ZuCo text + labels"),
    "zuco_standard": (
        "ZuCo_SST_data/combined_sst_et_standard.csv",
        "ZuCo text + z-scored gaze",
    ),
    "zuco_minmax": (
        "ZuCo_SST_data/combined_sst_et_min_max.csv",
        "ZuCo text + min-max gaze",
    ),
    "zuco_train": ("ZuCo_SST_data/train.csv", "ZuCo 80% split"),
    "zuco_valid": ("ZuCo_SST_data/valid.csv", "ZuCo 10% valid"),
    "zuco_test": ("ZuCo_SST_data/test.csv", "ZuCo 10% test"),
    "zuco_subject_1": ("ZuCo_et_csv_data/1_SR.csv", "ZuCo subject 1 sentences"),
    "zuco_average": ("ZuCo_et_csv_data/average_data.csv", "ZuCo subject-mean sentences"),
    "zuco_word_avg": (
        "ZuCo_et_csv_data/word/word_averages_v2.csv",
        "ZuCo word-level subject mean",
    ),
    "sst_combined": ("SST_data/combined_full_sst_et.csv", "Full SST + projected gaze"),
    "sst_train": ("SST_data/train_full_sst.csv", "Full SST train"),
    "sst_valid": ("SST_data/valid_full_sst.csv", "Full SST valid"),
    "sst_test": ("SST_data/test_full_sst.csv", "Full SST test"),
    "pred_word_small": (
        "gaze_prediction/data/prediction_test.csv",
        "Word-level predicted ET (small)",
    ),
}


SENTIMENT_TABLES = (
    "zuco_text",
    "zuco_standard",
    "zuco_minmax",
    "zuco_train",
    "zuco_valid",
    "zuco_test",
    "sst_combined",
    "sst_train",
    "sst_valid",
    "sst_test",
)

SPLIT_GROUPS = {
    "zuco": ("zuco_standard", "zuco_train", "zuco_valid", "zuco_test"),
    "sst": ("sst_combined", "sst_train", "sst_valid", "sst_test"),
}


def table_path(name: str) -> Path:
    rel, _role = TABLES[name]
    return repo_path(rel)


def table_role(name: str) -> str:
    return TABLES[name][1]
