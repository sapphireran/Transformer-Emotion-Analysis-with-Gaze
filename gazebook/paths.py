"""Resolve the checkout root from this file or the current working directory."""

from __future__ import annotations

from pathlib import Path

_MARKERS = ("model_ZuCo_SST.py", "ZuCo_et_csv_data", "SST_data")


def repo_root(start: Path | None = None) -> Path:
    """Return the repository root that contains the committed gaze tables.

    Walks parents of this package, then parents of ``start`` (default: cwd).
    """
    candidates = [Path(__file__).resolve().parent.parent]
    cursor = (start or Path.cwd()).resolve()
    candidates.append(cursor)
    candidates.extend(cursor.parents)
    for cand in candidates:
        if all((cand / marker).exists() for marker in _MARKERS):
            return cand
    raise FileNotFoundError(
        "Could not locate the Transformer-Emotion-Analysis-with-Gaze root. "
        "Run examples from the checkout or set the working directory there."
    )


def p(rel: str, start: Path | None = None) -> Path:
    """Join ``rel`` onto :func:`repo_root`."""
    return repo_root(start) / rel
