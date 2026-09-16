"""Locate the repository root from any working directory."""

from __future__ import annotations

from pathlib import Path

_MARKERS = ("model_full_SST.py", "model_ZuCo_SST.py", "ZuCo_SST_data", "SST_data")


def repo_root(start: Path | None = None) -> Path:
    """Walk parents until the original training scripts and data dirs appear."""
    here = Path(start).resolve() if start is not None else Path(__file__).resolve()
    candidates = [here, *here.parents]
    for path in candidates:
        if all((path / marker).exists() for marker in _MARKERS):
            return path
    raise FileNotFoundError(
        "Could not find the Transformer-Emotion-Analysis-with-Gaze root "
        f"(looked upward from {here})."
    )


def data_path(*parts: str) -> Path:
    return repo_root().joinpath(*parts)


def docs_assets() -> Path:
    path = repo_root() / "docs" / "assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def examples_output() -> Path:
    path = repo_root() / "examples" / "output"
    path.mkdir(parents=True, exist_ok=True)
    return path
