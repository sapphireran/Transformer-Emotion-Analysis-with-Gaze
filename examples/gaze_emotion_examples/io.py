"""CSV loading helpers that honor DatasetSpec column roles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .catalog import DatasetSpec, get_dataset

if TYPE_CHECKING:
    from pathlib import Path


def load_dataset(spec: DatasetSpec | str) -> pd.DataFrame:
    """Read a catalogued table and check that declared columns exist."""
    if isinstance(spec, str):
        spec = get_dataset(spec)
    if not spec.path.exists():
        raise FileNotFoundError(f"Dataset file is missing: {spec.path}")
    frame = pd.read_csv(spec.path)
    _assert_columns(frame, spec)
    return frame


def load_columns(spec: DatasetSpec | str, columns: list[str]) -> pd.DataFrame:
    """Read only the requested columns from a catalogued table."""
    frame = load_dataset(spec)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise KeyError(f"Columns not in table: {missing}")
    return frame.loc[:, columns].copy()


def gaze_frame(spec: DatasetSpec | str) -> pd.DataFrame:
    """Return only the numeric gaze channels declared on the spec."""
    if isinstance(spec, str):
        spec = get_dataset(spec)
    if not spec.gaze_columns:
        raise ValueError(f"{spec.key} does not declare gaze columns")
    return load_columns(spec, list(spec.gaze_columns))


def _assert_columns(frame: pd.DataFrame, spec: DatasetSpec) -> None:
    expected = [
        spec.label_column,
        spec.id_column,
        spec.text_column,
        *spec.gaze_columns,
    ]
    missing = [name for name in expected if name and name not in frame.columns]
    if missing:
        raise ValueError(
            f"{spec.relative_path} is missing declared columns: {missing}. "
            f"Available: {list(frame.columns)}"
        )


def write_csv(frame: pd.DataFrame, path: "Path") -> None:
    """Write a CSV, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
