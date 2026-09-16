"""Resolve repo-root paths without depending on the process cwd."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (parent of ``examples/``)."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RepoPaths:
    root: Path
    zuco_sst_dir: Path
    zuco_et_dir: Path
    sst_dir: Path
    gaze_pred_dir: Path
    result_dir: Path

    @property
    def zuco_combined_standard(self) -> Path:
        return self.zuco_sst_dir / "combined_sst_et_standard.csv"

    @property
    def zuco_combined_minmax(self) -> Path:
        return self.zuco_sst_dir / "combined_sst_et_min_max.csv"

    @property
    def zuco_text(self) -> Path:
        return self.zuco_sst_dir / "ssts_ZuCo.csv"

    @property
    def zuco_train(self) -> Path:
        return self.zuco_sst_dir / "train.csv"

    @property
    def zuco_valid(self) -> Path:
        return self.zuco_sst_dir / "valid.csv"

    @property
    def zuco_test(self) -> Path:
        return self.zuco_sst_dir / "test.csv"

    @property
    def zuco_average(self) -> Path:
        return self.zuco_et_dir / "average_data.csv"

    @property
    def zuco_average_standard(self) -> Path:
        return self.zuco_et_dir / "standard_scaled_average_data.csv"

    @property
    def word_averages(self) -> Path:
        return self.zuco_et_dir / "word" / "word_averages_v2.csv"

    @property
    def full_sst_train(self) -> Path:
        return self.sst_dir / "train_full_sst.csv"

    @property
    def full_sst_valid(self) -> Path:
        return self.sst_dir / "valid_full_sst.csv"

    @property
    def full_sst_test(self) -> Path:
        return self.sst_dir / "test_full_sst.csv"

    @property
    def full_sst_text(self) -> Path:
        return self.sst_dir / "stts_all_sentence_level.csv"

    @property
    def predicted_gaze_v2(self) -> Path:
        return self.gaze_pred_dir / "prediction_test_v2.csv"

    def subject_sentence_et(self, subject_one_based: int) -> Path:
        if subject_one_based < 1 or subject_one_based > 12:
            raise ValueError("subject_one_based must be in 1..12")
        return self.zuco_et_dir / f"{subject_one_based}_SR.csv"


def default_paths(root: Path | None = None) -> RepoPaths:
    base = Path(root) if root is not None else repo_root()
    return RepoPaths(
        root=base,
        zuco_sst_dir=base / "ZuCo_SST_data",
        zuco_et_dir=base / "ZuCo_et_csv_data",
        sst_dir=base / "SST_data",
        gaze_pred_dir=base / "gaze_prediction" / "data",
        result_dir=base / "result",
    )
