"""Lightweight table profiling for the documentation scripts."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .schema import LABEL_NAMES


def label_counts(series: pd.Series) -> dict[int, int]:
    counts = {int(k): int(v) for k, v in series.value_counts().sort_index().items()}
    return counts


def label_shares(series: pd.Series) -> dict[str, float]:
    n = max(len(series), 1)
    return {LABEL_NAMES[k]: v / n for k, v in label_counts(series).items()}


def frame_profile(df: pd.DataFrame, name: str) -> dict[str, Any]:
    numeric = df.select_dtypes(include="number")
    profile: dict[str, Any] = {
        "name": name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "n_numeric": int(numeric.shape[1]),
        "n_missing": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if "sentiment_label" in df.columns:
        profile["label_counts"] = label_counts(df["sentiment_label"])
        profile["label_shares"] = {
            k: round(v, 4) for k, v in label_shares(df["sentiment_label"]).items()
        }
    if "sentence" in df.columns:
        lengths = df["sentence"].astype(str).str.len()
        profile["sentence_chars"] = {
            "mean": float(lengths.mean()),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
        }
    if "sentence_id" in df.columns:
        profile["n_sentence_id"] = int(df["sentence_id"].nunique())
        profile["sentence_id_min"] = int(df["sentence_id"].min())
        profile["sentence_id_max"] = int(df["sentence_id"].max())
    moments: dict[str, dict[str, float]] = {}
    for col in numeric.columns:
        if col in {"sentence_id", "id", "sentiment_label", "word_id", "Word_ID"}:
            continue
        ser = numeric[col]
        moments[col] = {
            "mean": float(ser.mean()),
            "std": float(ser.std(ddof=1)) if len(ser) > 1 else 0.0,
            "min": float(ser.min()),
            "max": float(ser.max()),
        }
    profile["moments"] = moments
    return profile


def profiles_to_markdown(profiles: list[dict[str, Any]]) -> str:
    lines = [
        "| Table | Rows | Missing | Labels (0/1/2) |",
        "| --- | ---: | ---: | --- |",
    ]
    for p in profiles:
        labels = p.get("label_counts", {})
        label_s = (
            f"{labels.get(0, '—')} / {labels.get(1, '—')} / {labels.get(2, '—')}"
            if labels
            else "—"
        )
        lines.append(
            f"| `{p['name']}` | {p['rows']} | {p['n_missing']} | {label_s} |"
        )
    return "\n".join(lines) + "\n"
