"""Repo-relative paths for the CSVs that already live in this personal project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def detect_repo_root(start: Path | None = None) -> Path:
    """Walk parents until the original training scripts and data dirs are found."""
    here = (start or Path(__file__).resolve()).resolve()
    candidates = [here, *here.parents] if here.is_file() else [here, *here.parents]
    for path in candidates:
        if (path / "model_ZuCo_SST.py").is_file() and (path / "ZuCo_SST_data").is_dir():
            return path
    cwd = Path.cwd()
    if (cwd / "model_ZuCo_SST.py").is_file():
        return cwd
    raise FileNotFoundError(
        "Could not locate the personal Transformer-Emotion-Analysis-with-Gaze root."
    )


@dataclass(frozen=True)
class DataPaths:
    """All checked-in tables used by docs and examples."""

    root: Path

    @classmethod
    def from_cwd(cls, start: Path | None = None) -> "DataPaths":
        return cls(root=detect_repo_root(start))

    @property
    def zuco_sst(self) -> Path:
        return self.root / "ZuCo_SST_data"

    @property
    def sst(self) -> Path:
        return self.root / "SST_data"

    @property
    def zuco_sentence_et(self) -> Path:
        return self.root / "ZuCo_et_csv_data"

    @property
    def zuco_word_et(self) -> Path:
        return self.root / "ZuCo_et_csv_data" / "word"

    @property
    def gaze_prediction(self) -> Path:
        return self.root / "gaze_prediction" / "data"

    @property
    def zuco_text(self) -> Path:
        return self.zuco_sst / "ssts_ZuCo.csv"

    @property
    def zuco_combined_standard(self) -> Path:
        return self.zuco_sst / "combined_sst_et_standard.csv"

    @property
    def zuco_combined_minmax(self) -> Path:
        return self.zuco_sst / "combined_sst_et_min_max.csv"

    @property
    def zuco_train(self) -> Path:
        return self.zuco_sst / "train.csv"

    @property
    def zuco_valid(self) -> Path:
        return self.zuco_sst / "valid.csv"

    @property
    def zuco_test(self) -> Path:
        return self.zuco_sst / "test.csv"

    @property
    def full_sst_combined(self) -> Path:
        return self.sst / "combined_full_sst_et.csv"

    @property
    def full_sst_train(self) -> Path:
        return self.sst / "train_full_sst.csv"

    @property
    def full_sst_valid(self) -> Path:
        return self.sst / "valid_full_sst.csv"

    @property
    def full_sst_test(self) -> Path:
        return self.sst / "test_full_sst.csv"

    @property
    def word_averages(self) -> Path:
        return self.zuco_word_et / "word_averages_v2.csv"

    @property
    def sentence_averages_raw(self) -> Path:
        return self.zuco_sentence_et / "average_data.csv"

    def subject_sentence_csv(self, subject: int) -> Path:
        if subject < 1 or subject > 12:
            raise ValueError("ZuCo subjects in this export are numbered 1-12")
        return self.zuco_sentence_et / f"{subject}_SR.csv"

    def subject_word_csv(self, subject: int) -> Path:
        if subject < 1 or subject > 12:
            raise ValueError("ZuCo subjects in this export are numbered 1-12")
        return self.zuco_word_et / f"{subject}_SR.csv"
