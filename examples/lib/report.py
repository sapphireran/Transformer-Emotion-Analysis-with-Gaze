"""Plain-text and markdown tables for the example scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence


def pad(cell: object) -> str:
    if isinstance(cell, float):
        return f"{cell:.4f}"
    return str(cell)


def ascii_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    grid = [list(headers)] + [[pad(c) for c in row] for row in rows]
    widths = [max(len(str(grid[r][c])) for r in range(len(grid))) for c in range(len(headers))]

    def fmt(row: Sequence[object]) -> str:
        return "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    rule = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    lines = [fmt(grid[0]), rule]
    lines.extend(fmt(row) for row in grid[1:])
    return "\n".join(lines)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    return ascii_table(headers, rows)


def bullet(lines: Iterable[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.rstrip() + "\n", encoding="utf-8")


def join_sections(parts: Sequence[str]) -> str:
    return "\n\n".join(part.rstrip() for part in parts if part.strip()) + "\n"


def format_confusion(matrix: Sequence[Sequence[int]], labels: Sequence[str]) -> str:
    headers = [""] + [f"pred {lab}" for lab in labels]
    rows: List[List[object]] = []
    for name, row in zip(labels, matrix):
        rows.append([f"true {name}", *row])
    return ascii_table(headers, rows)
