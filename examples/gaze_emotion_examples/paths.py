"""Locate the repository root from any nested example or test file."""

from __future__ import annotations

from pathlib import Path

_MARKERS = ("model_ZuCo_SST.py", "model_full_SST.py", "ZuCo_SST_data")


def repo_root(start: Path | None = None) -> Path:
    """Walk upward until the original training scripts and data dirs appear."""
    here = (start or Path(__file__)).resolve()
    if here.is_file():
        here = here.parent
    for candidate in (here, *here.parents):
        if all((candidate / marker).exists() for marker in _MARKERS):
            return candidate
    raise RuntimeError(
        f"Could not find the Transformer-Emotion-Analysis-with-Gaze root from {here}"
    )


def data_path(*parts: str) -> Path:
    """Join path parts onto the repository root."""
    return repo_root().joinpath(*parts)
