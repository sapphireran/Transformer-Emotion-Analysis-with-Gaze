"""Gaze Sidecar Atlas — personal helpers for this repo's two experiment tracks.

The training models treat eye-tracking as a *sidecar*: five scalar gaze
features are linearly projected to 16 dimensions and concatenated with a
transformer pooler vector (768-d). Nothing in this package downloads BERT
or RoBERTa; it only inspects the tables already in the checkout and
reproduces the fusion *geometry* with NumPy.
"""

from .alignment import (
    SUBJECT3_1INDEX,
    SUBJECT3_0INDEX,
    TASK1_SKIPPED_ORIGINALS,
    pack_subject3_id,
    unpack_subject3_id,
    subject3_sentence_alignment,
    word_break_row,
)
from .fusion import SidecarFusion, FUSION_CONCAT_DIM
from .metrics import (
    last_batch_predictions,
    full_predictions,
    weighted_scores,
    majority_baseline,
)
from .paths import ROOT, expected_tables
from .rank import gaze_pca, collinearity_matrix

__all__ = [
    "SUBJECT3_1INDEX",
    "SUBJECT3_0INDEX",
    "TASK1_SKIPPED_ORIGINALS",
    "pack_subject3_id",
    "unpack_subject3_id",
    "subject3_sentence_alignment",
    "word_break_row",
    "SidecarFusion",
    "FUSION_CONCAT_DIM",
    "last_batch_predictions",
    "full_predictions",
    "weighted_scores",
    "majority_baseline",
    "ROOT",
    "expected_tables",
    "gaze_pca",
    "collinearity_matrix",
]
