"""Lightweight helpers for inspecting the personal ZuCo / SST gaze tables.

This package is intentionally independent of torch and transformers. It reads
the CSVs that already live in the repo and exposes the same five gaze channels
the fusion models consume.
"""

from .schema import (
    CANONICAL_GAZE,
    EXTRA_SENTENCE_ET,
    LABEL_NAMES,
    SENTIMENT_FROM_INT,
    SENTIMENT_TO_INT,
    ZUCO_TO_CANONICAL,
)
from .paths import default_paths, repo_root

__all__ = [
    "CANONICAL_GAZE",
    "EXTRA_SENTENCE_ET",
    "LABEL_NAMES",
    "SENTIMENT_FROM_INT",
    "SENTIMENT_TO_INT",
    "ZUCO_TO_CANONICAL",
    "default_paths",
    "repo_root",
]

__version__ = "0.1.0"
