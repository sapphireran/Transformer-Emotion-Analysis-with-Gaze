"""Pretty-print helpers shared by the example scripts."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .baselines import BaselineReport
from .splits import SplitReport


def _hr(title: str) -> str:
    bar = "=" * 72
    return f"{bar}\n{title}\n{bar}"


def format_frame(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    if df.empty:
        return "(empty)"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: floatfmt.format(x) if pd.notna(x) else "")
    return out.to_string(index=False)


def format_baseline(report: BaselineReport) -> str:
    lines = [
        f"{report.name}: features={list(report.feature_names)}",
        (
            f"  acc {report.mean_accuracy:.4f} ± {report.std_accuracy:.4f}  |  "
            f"weighted F1 {report.mean_f1_weighted:.4f} ± {report.std_f1_weighted:.4f}  |  "
            f"macro F1 {report.mean_f1_macro:.4f} ± {report.std_f1_macro:.4f}"
        ),
    ]
    for fold in report.folds:
        m = fold.metrics
        lines.append(
            f"    fold {fold.fold}: acc={m.accuracy:.4f}  "
            f"wF1={m.f1_weighted:.4f}  mF1={m.f1_macro:.4f}  n={m.n}"
        )
    return "\n".join(lines)


def format_split(report: SplitReport) -> str:
    lines = [
        f"sizes: train={report.n_train}  valid={report.n_valid}  test={report.n_test}",
        f"id column: {report.id_column}",
        f"disjoint: {report.is_disjoint}",
        "overlaps:",
    ]
    for key, ids in report.overlapping_ids.items():
        preview = ids[:8]
        extra = f" … (+{len(ids) - 8})" if len(ids) > 8 else ""
        lines.append(f"  {key}: {len(ids)} {preview}{extra}")
    lines.append("label fractions:")
    lines.append(format_frame(report.label_fractions.reset_index().rename(columns={"index": "label"})))
    if report.notes:
        lines.append("notes:")
        lines.extend(f"  - {n}" for n in report.notes)
    return "\n".join(lines)


def section(title: str, body: Any) -> str:
    return f"{_hr(title)}\n{body}\n"
