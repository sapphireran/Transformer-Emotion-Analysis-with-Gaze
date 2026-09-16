"""Shared fixtures: always resolve data from the git checkout."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from teag_examples.paths import repo_root  # noqa: E402


@pytest.fixture(scope="session")
def root() -> Path:
    return repo_root()
