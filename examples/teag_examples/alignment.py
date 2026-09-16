"""Subject-3 compacted index vs original ZuCo Task 1 sentence index."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .io import load_subject_sentence_gaze
from .schema import SUBJECT_3_SKIP_ORIGINAL, ZUCO_N_SENTENCES, ZUCO_SUBJECT_3_N_SENTENCES

# Original indices skipped for Task 1 subject index 2 (file 3_SR.csv).
SUBJECT_3_SKIP_SET = frozenset(SUBJECT_3_SKIP_ORIGINAL)


def original_to_compacted_subject3(original: int) -> int | None:
    """Map original sentence index 0–399 → compacted id, or None if skipped."""
    if original in SUBJECT_3_SKIP_SET:
        return None
    if original < 0 or original >= ZUCO_N_SENTENCES:
        raise ValueError(original)
    skipped_before = sum(1 for s in SUBJECT_3_SKIP_ORIGINAL if s < original)
    return original - skipped_before


def compacted_to_original_subject3(compacted: int) -> int:
    """Inverse of :func:`original_to_compacted_subject3`."""
    if compacted < 0 or compacted >= ZUCO_SUBJECT_3_N_SENTENCES:
        raise ValueError(compacted)
    if compacted < 150:
        return compacted
    # 100 originals (150–249) were dropped; 399 is dropped after 398.
    return compacted + 100


def first_sentlen_mismatch(
    reference: pd.DataFrame, other: pd.DataFrame, column: str = "SentLen"
) -> int | None:
    """First row where ``SentLen`` disagrees, or None if all compared rows match."""
    n = min(len(reference), len(other))
    ref = reference[column].to_numpy()[:n]
    oth = other[column].to_numpy()[:n]
    neq = np.where(ref != oth)[0]
    if neq.size == 0:
        return None
    return int(neq[0])


def subject3_alignment_report(
    subject3: pd.DataFrame | None = None,
    reference: pd.DataFrame | None = None,
) -> dict:
    subj3 = subject3 if subject3 is not None else load_subject_sentence_gaze(3)
    ref = reference if reference is not None else load_subject_sentence_gaze(1)
    mismatch = first_sentlen_mismatch(ref, subj3)
    prefix = mismatch if mismatch is not None else min(len(ref), len(subj3))
    # After the skip, compacted 150 should equal reference original 250
    shifted = False
    if len(subj3) > 150 and len(ref) > 260:
        shifted = bool(
            np.array_equal(
                subj3["SentLen"].to_numpy()[150:161],
                ref["SentLen"].to_numpy()[250:261],
            )
        )
    return {
        "n_subject3": int(len(subj3)),
        "n_reference": int(len(ref)),
        "first_mismatch_row": mismatch,
        "aligned_prefix_rows": int(prefix),
        "compacted_150_matches_original_250": shifted,
        "n_skipped_original": len(SUBJECT_3_SKIP_ORIGINAL),
    }


def skip_list_matches_transformer(skip: Sequence[int] = SUBJECT_3_SKIP_ORIGINAL) -> bool:
    """Sanity: 101 skipped originals → 299 remaining rows."""
    return ZUCO_N_SENTENCES - len(set(skip)) == ZUCO_SUBJECT_3_N_SENTENCES
