from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from gaze_emotion_examples.paths import repo_root  # noqa: E402


@pytest.fixture(scope="session")
def root() -> Path:
    return repo_root()
