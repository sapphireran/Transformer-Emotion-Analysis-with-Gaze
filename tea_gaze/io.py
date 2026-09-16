"""CSV loaders that check the columns the training scripts actually read."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .paths import DataPaths, detect_repo_root
from .schema import required_columns


def repo_root(start: Path | None = None) -> Path:
    return detect_repo_root(start)


def load_csv(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} is empty")
    return frame


def _require(frame: pd.DataFrame, kind: str, path: Path) -> pd.DataFrame:
    missing = [col for col in required_columns(kind) if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing} for schema {kind!r}")
    return frame


@dataclass(frozen=True)
class DatasetBundle:
    """A named table plus the schema kind used to validate it."""

    name: str
    kind: str
    path: Path
    frame: pd.DataFrame

    @property
    def n_rows(self) -> int:
        return int(len(self.frame))

    @property
    def columns(self) -> list[str]:
        return list(self.frame.columns)


def load_zuco_sentiment(paths: DataPaths | None = None) -> DatasetBundle:
    paths = paths or DataPaths.from_cwd()
    path = paths.zuco_text
    frame = _require(load_csv(path), "zuco_text", path)
    return DatasetBundle("zuco_text", "zuco_text", path, frame)


def load_zuco_combined(paths: DataPaths | None = None, scaling: str = "standard") -> DatasetBundle:
    paths = paths or DataPaths.from_cwd()
    path = paths.zuco_combined_standard if scaling == "standard" else paths.zuco_combined_minmax
    if scaling not in {"standard", "minmax"}:
        raise ValueError("scaling must be 'standard' or 'minmax'")
    frame = _require(load_csv(path), "zuco_sentence_et", path)
    return DatasetBundle(f"zuco_{scaling}", "zuco_sentence_et", path, frame)


def load_zuco_splits(paths: DataPaths | None = None) -> dict[str, DatasetBundle]:
    paths = paths or DataPaths.from_cwd()
    out: dict[str, DatasetBundle] = {}
    for name, path in {
        "train": paths.zuco_train,
        "valid": paths.zuco_valid,
        "test": paths.zuco_test,
    }.items():
        frame = _require(load_csv(path), "zuco_sentence_et", path)
        out[name] = DatasetBundle(f"zuco_{name}", "zuco_sentence_et", path, frame)
    return out


def load_full_sst_splits(paths: DataPaths | None = None) -> dict[str, DatasetBundle]:
    paths = paths or DataPaths.from_cwd()
    mapping = {
        "combined": paths.full_sst_combined,
        "train": paths.full_sst_train,
        "valid": paths.full_sst_valid,
        "test": paths.full_sst_test,
    }
    out: dict[str, DatasetBundle] = {}
    for name, path in mapping.items():
        frame = _require(load_csv(path), "full_sst", path)
        out[name] = DatasetBundle(f"full_sst_{name}", "full_sst", path, frame)
    return out


def load_word_averages(paths: DataPaths | None = None) -> DatasetBundle:
    paths = paths or DataPaths.from_cwd()
    path = paths.word_averages
    frame = _require(load_csv(path), "zuco_word", path)
    return DatasetBundle("zuco_word_averages", "zuco_word", path, frame)


def load_subject_sentence_et(subject: int, paths: DataPaths | None = None) -> DatasetBundle:
    paths = paths or DataPaths.from_cwd()
    path = paths.subject_sentence_csv(subject)
    frame = load_csv(path)
    return DatasetBundle(f"subject_{subject}_sentence", "subject_sentence", path, frame)


def load_gaze_prediction_sample(paths: DataPaths | None = None) -> DatasetBundle:
    paths = paths or DataPaths.from_cwd()
    path = paths.gaze_prediction / "prediction_test.csv"
    frame = load_csv(path)
    return DatasetBundle("gaze_prediction_test", "predicted_word", path, frame)
