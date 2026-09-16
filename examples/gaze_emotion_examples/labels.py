"""SST-3 label helpers used across the examples."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

LABEL_NAMES: dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

LABEL_FROM_NAME: dict[str, int] = {
    "NEGATIVE": 0,
    "NEUTRAL": 1,
    "POSITIVE": 2,
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


def label_name(code: int) -> str:
    """Map a numeric SST-3 code to a readable name."""
    try:
        return LABEL_NAMES[int(code)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unsupported sentiment code: {code!r}") from exc


def label_code(name: str) -> int:
    """Map a readable or original-folder name to the numeric SST-3 code."""
    try:
        return LABEL_FROM_NAME[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported sentiment name: {name!r}") from exc


def label_counts(series: pd.Series) -> pd.DataFrame:
    """Return counts, percents, and names for a label column."""
    counts = series.value_counts(dropna=False).sort_index()
    total = int(counts.sum())
    rows = []
    for code, count in counts.items():
        name = label_name(int(code)) if pd.notna(code) and int(code) in LABEL_NAMES else str(code)
        rows.append(
            {
                "label": int(code) if pd.notna(code) else None,
                "name": name,
                "count": int(count),
                "percent": (100.0 * count / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def majority_accuracy(series: pd.Series) -> float:
    """Accuracy of always predicting the most common label."""
    if series.empty:
        return 0.0
    return float(series.value_counts(normalize=True).iloc[0])


def label_table_markdown(frame: pd.DataFrame, title: str | None = None) -> str:
    """Render a counts table as a GitHub-flavored markdown table."""
    lines = []
    if title:
        lines.append(f"### {title}")
        lines.append("")
    lines.append("| label | name | count | percent |")
    lines.append("| ---: | --- | ---: | ---: |")
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.label} | {row.name} | {row.count} | {row.percent:.1f}% |"
        )
    return "\n".join(lines)


def as_named_mapping(counts: Mapping[int, int]) -> dict[str, int]:
    """Convert ``{0: n}`` counts into ``{"negative": n}``."""
    return {label_name(code): int(count) for code, count in counts.items()}
