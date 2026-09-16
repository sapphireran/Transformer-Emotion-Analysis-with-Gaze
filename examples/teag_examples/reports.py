"""Write markdown / JSON summaries used by the example scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(payload: Any, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return dest


def write_markdown(text: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return dest


def corr_to_markdown(corr) -> str:
    cols = list(corr.columns)
    header = "|  | " + " | ".join(f"`{c}`" for c in cols) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in cols) + " |"
    rows = [header, sep]
    for idx in corr.index:
        cells = [f"`{idx}`"] + [f"{corr.loc[idx, c]:.3f}" for c in cols]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"
