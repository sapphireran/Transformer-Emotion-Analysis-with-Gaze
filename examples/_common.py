"""Tiny shared CLI helpers for the numbered example scripts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
