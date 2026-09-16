"""Plain-text tables for the example scripts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from .schema import SENTIMENT_ID_TO_NAME, label_name


def _fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(int(value))
    return str(value)


def markdown_table(frame: pd.DataFrame, digits: int = 4, max_rows: int | None = None) -> str:
    view = frame if max_rows is None else frame.head(max_rows)
    headers = [str(col) for col in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in view.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(cell, digits) for cell in row) + " |")
    return "\n".join(lines)


def label_counts(frame: pd.DataFrame, label_column: str = "sentiment_label") -> pd.DataFrame:
    counts = frame[label_column].value_counts().sort_index()
    rows = []
    total = int(counts.sum())
    for label, count in counts.items():
        rows.append(
            {
                "label": int(label),
                "name": label_name(int(label)),
                "count": int(count),
                "share": float(count) / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def schema_rows(name: str, path: str, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "name": name,
        "rows": int(len(frame)),
        "cols": int(frame.shape[1]),
        "path": path,
        "columns": ", ".join(frame.columns),
    }


def series_to_frame(series: pd.Series, value_name: str = "value") -> pd.DataFrame:
    return series.rename(value_name).rename_axis("feature").reset_index()


def glossary_frame(keys: Iterable[str], glossary: Mapping[str, str]) -> pd.DataFrame:
    rows = []
    for key in keys:
        rows.append({"feature": key, "meaning": glossary.get(key, "(not in glossary)")})
    return pd.DataFrame(rows)


def sentiment_legend() -> str:
    return ", ".join(f"{idx}={name}" for idx, name in SENTIMENT_ID_TO_NAME.items())
