"""Personal lab helpers for this ZuCo + SST gaze-sentiment checkout.

The original trainers stay untouched. This package only reads the committed
CSVs so notes and examples can be regenerated on a CPU box without
downloading BERT/RoBERTa weights.
"""

from .paths import repo_root
from .csvio import read_dicts, read_headerless, float_col, table_to_array
from . import schema, remap, stats, metrics, tokens, baselines

__all__ = [
    "repo_root",
    "read_dicts",
    "read_headerless",
    "float_col",
    "table_to_array",
    "schema",
    "remap",
    "stats",
    "metrics",
    "tokens",
    "baselines",
]
