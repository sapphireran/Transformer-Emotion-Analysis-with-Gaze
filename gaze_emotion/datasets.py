"""CSV loaders for the checked-in ZuCo and full-SST tables."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .constants import (
    FUSION_GAZE_FEATURES,
    FUSION_GAZE_FEATURES_FULL_SST,
    LABEL_ID_TO_NAME,
    fusion_feature_names,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DatasetSummary:
    path: str
    n_rows: int
    columns: tuple[str, ...]
    label_counts: dict[str, int]
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    missing_by_column: dict[str, int]

    def as_lines(self) -> list[str]:
        lines = [
            f"file: {self.path}",
            f"rows: {self.n_rows}",
            f"columns ({len(self.columns)}): {', '.join(self.columns)}",
            "label counts:",
        ]
        for name, count in sorted(self.label_counts.items()):
            pct = 100.0 * count / self.n_rows if self.n_rows else 0.0
            lines.append(f"  {name}: {count} ({pct:.1f}%)")
        if self.feature_means:
            lines.append("feature means / stds:")
            for name in self.feature_means:
                lines.append(
                    f"  {name}: mean={self.feature_means[name]:.4f} "
                    f"std={self.feature_stds[name]:.4f}"
                )
        missing = {k: v for k, v in self.missing_by_column.items() if v}
        if missing:
            lines.append("missing / empty cells:")
            for name, count in missing.items():
                lines.append(f"  {name}: {count}")
        return lines

    def __str__(self) -> str:
        return "\n".join(self.as_lines())


def resolve_data_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    anchored = REPO_ROOT / candidate
    if anchored.is_file():
        return anchored
    raise FileNotFoundError(f"Could not find data file: {path}")


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Load a headered CSV as a list of string dictionaries."""
    csv_path = resolve_data_path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def feature_matrix(
    rows: Sequence[dict[str, str]],
    columns: Sequence[str],
) -> np.ndarray:
    """Build a float matrix from named CSV columns."""
    missing = [col for col in columns if rows and col not in rows[0]]
    if missing:
        raise KeyError(f"Missing columns {missing} in {list(rows[0])}")
    data = [[_to_float(row[col]) for col in columns] for row in rows]
    return np.asarray(data, dtype=float)


def infer_fusion_columns(columns: Iterable[str]) -> tuple[str, ...]:
    cols = set(columns)
    if set(FUSION_GAZE_FEATURES).issubset(cols):
        return FUSION_GAZE_FEATURES
    if set(FUSION_GAZE_FEATURES_FULL_SST).issubset(cols):
        return FUSION_GAZE_FEATURES_FULL_SST
    raise KeyError(
        "Could not find a fusion feature set. "
        f"Looked for {FUSION_GAZE_FEATURES} or {FUSION_GAZE_FEATURES_FULL_SST}."
    )


def summarize_rows(
    rows: Sequence[dict[str, str]],
    path: str | Path = "",
    label_column: str = "sentiment_label",
    feature_columns: Sequence[str] | None = None,
) -> DatasetSummary:
    columns = tuple(rows[0].keys()) if rows else tuple()
    counts: Counter[str] = Counter()
    if label_column and rows and label_column in rows[0]:
        for row in rows:
            raw = row[label_column]
            try:
                counts[LABEL_ID_TO_NAME[int(raw)]] += 1
            except (KeyError, ValueError):
                counts[str(raw)] += 1

    if feature_columns is None:
        try:
            feature_columns = infer_fusion_columns(columns)
        except KeyError:
            feature_columns = tuple()

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    if feature_columns and rows:
        matrix = feature_matrix(rows, feature_columns)
        for idx, name in enumerate(feature_columns):
            col = matrix[:, idx]
            means[name] = float(np.nanmean(col))
            stds[name] = float(np.nanstd(col))

    missing: dict[str, int] = {}
    for name in columns:
        missing[name] = sum(1 for row in rows if row.get(name, "") in ("", None))

    return DatasetSummary(
        path=str(path),
        n_rows=len(rows),
        columns=columns,
        label_counts=dict(counts),
        feature_means=means,
        feature_stds=stds,
        missing_by_column=missing,
    )


def load_and_summarize(path: str | Path, **kwargs) -> DatasetSummary:
    rows = load_csv_rows(path)
    return summarize_rows(rows, path=path, **kwargs)


def sentence_ids(rows: Sequence[dict[str, str]], column: str = "sentence_id") -> set[str]:
    return {row[column] for row in rows if column in row}


def split_overlap(left: Sequence[dict[str, str]], right: Sequence[dict[str, str]]) -> set[str]:
    return sentence_ids(left) & sentence_ids(right)


KNOWN_TABLES: dict[str, str] = {
    "zuco_combined_standard": "ZuCo_SST_data/combined_sst_et_standard.csv",
    "zuco_combined_minmax": "ZuCo_SST_data/combined_sst_et_min_max.csv",
    "zuco_train": "ZuCo_SST_data/train.csv",
    "zuco_valid": "ZuCo_SST_data/valid.csv",
    "zuco_test": "ZuCo_SST_data/test.csv",
    "zuco_text_only": "ZuCo_SST_data/ssts_ZuCo.csv",
    "full_sst_combined": "SST_data/combined_full_sst_et.csv",
    "full_sst_train": "SST_data/train_full_sst.csv",
    "full_sst_valid": "SST_data/valid_full_sst.csv",
    "full_sst_test": "SST_data/test_full_sst.csv",
    "zuco_sentence_average": "ZuCo_et_csv_data/average_data.csv",
    "zuco_word_average": "ZuCo_et_csv_data/word/word_averages_v2.csv",
}


def known_table(name: str) -> Path:
    try:
        return resolve_data_path(KNOWN_TABLES[name])
    except KeyError as exc:
        raise KeyError(f"Unknown table {name!r}. Options: {sorted(KNOWN_TABLES)}") from exc


def fusion_style_for_path(path: str | Path) -> str:
    name = Path(path).as_posix()
    if "SST_data" in name and "ZuCo" not in name:
        return "full_sst"
    return "zuco"


# Re-export for callers that want the helper without importing constants.
__all__ = [
    "DatasetSummary",
    "KNOWN_TABLES",
    "REPO_ROOT",
    "feature_matrix",
    "fusion_feature_names",
    "fusion_style_for_path",
    "infer_fusion_columns",
    "known_table",
    "load_and_summarize",
    "load_csv_rows",
    "resolve_data_path",
    "sentence_ids",
    "split_overlap",
    "summarize_rows",
]
