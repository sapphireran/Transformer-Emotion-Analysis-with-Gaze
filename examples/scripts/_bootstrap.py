"""Put the ``examples/`` directory on ``sys.path`` so scripts can import teag_examples."""

from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_ROOT))
