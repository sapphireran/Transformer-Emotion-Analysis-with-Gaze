"""Small printing helpers so the numbered scripts look the same."""

from __future__ import annotations

from typing import Any

import pandas as pd


def banner(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}", flush=True)


def print_frame(df: pd.DataFrame, *, max_rows: int = 30, floatfmt: str = ".4f") -> None:
    with pd.option_context(
        "display.max_rows",
        max_rows,
        "display.max_columns",
        None,
        "display.width",
        120,
        "display.float_format",
        lambda v: format(v, floatfmt),
    ):
        print(df.to_string())


def json_ready(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value
