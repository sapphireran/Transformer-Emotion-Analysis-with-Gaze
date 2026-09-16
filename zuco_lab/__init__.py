"""Personal lab helpers for the ZuCo + SST gaze-sentiment workspace.

The original training scripts stay untouched. This package is a stdlib-only
layer for inventory, schema checks, subject-alignment audits, and tiny CPU
demos. Nothing here downloads BERT/RoBERTa or writes into ``models/``.
"""

from .paths import ROOT, DATASETS
from .schema import SchemaError, validate_rows
from .labels import LABEL_NAMES, label_name

__all__ = [
    "ROOT",
    "DATASETS",
    "SchemaError",
    "validate_rows",
    "LABEL_NAMES",
    "label_name",
    "__version__",
]

__version__ = "0.2.0"
