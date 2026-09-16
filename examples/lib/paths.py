"""Resolve the repository root from any working directory."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repo root (parent of ``examples/``)."""
    return Path(__file__).resolve().parents[2]


def resolve_root(root: str | Path | None = None) -> Path:
    """Use ``root`` if given, otherwise the repo that contains this file."""
    if root is None:
        return repo_root()
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"repository root does not exist: {path}")
    return path
