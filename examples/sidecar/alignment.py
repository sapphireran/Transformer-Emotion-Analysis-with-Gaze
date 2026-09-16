"""Subject-3 (1-indexed) / MATLAB subject 2 packed-index reconstruction.

``utils_ZuCo.DataTransformer`` skips Task-1 sentences 150-249 and 399 for
subject index 2, then writes a *new* contiguous ``id`` / ``Sent_ID``. The
committed CSVs therefore look like subject 3 is simply "shorter", but
row 150 is original sentence 250. Averaging subjects with
``groupby(level=0)`` silently mixes different movie-review sentences after
the hole.

This module reconstructs the packing so the atlas examples can *prove*
the misalignment from the committed tables (no MATLAB files required).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# MATLAB loop uses subject in range(12); CSV files are 1-indexed.
SUBJECT3_0INDEX = 2
SUBJECT3_1INDEX = 3

# Inclusive original sentence indices dropped for Task 1 / subject 2.
TASK1_SKIPPED_ORIGINALS: tuple[int, ...] = tuple(list(range(150, 250)) + [399])
N_ORIGINAL_SENTENCES = 400
N_PACKED_SENTENCES = N_ORIGINAL_SENTENCES - len(TASK1_SKIPPED_ORIGINALS)  # 299
PACK_SHIFT_AFTER = 150  # packed id >= 150 maps to original id + 100
PACK_SHIFT = 100  # skipped block length 150-249


def unpack_subject3_id(packed_id: int) -> int:
    """Map the committed subject-3 ``id`` back onto the 0-399 sentence grid."""
    if packed_id < 0 or packed_id >= N_PACKED_SENTENCES:
        raise ValueError(f"packed id {packed_id} is outside 0..{N_PACKED_SENTENCES - 1}")
    if packed_id < PACK_SHIFT_AFTER:
        return packed_id
    return packed_id + PACK_SHIFT


def pack_subject3_id(original_id: int) -> Optional[int]:
    """Inverse of ``unpack_subject3_id``. Returns None if the sentence was skipped."""
    if original_id < 0 or original_id >= N_ORIGINAL_SENTENCES:
        raise ValueError(f"original id {original_id} is outside 0..399")
    if original_id in TASK1_SKIPPED_ORIGINALS:
        return None
    if original_id < PACK_SHIFT_AFTER:
        return original_id
    return original_id - PACK_SHIFT


def subject3_sentence_alignment(
    subject1: pd.DataFrame,
    subject3: pd.DataFrame,
) -> pd.DataFrame:
    """Join subject 1 (complete 400) to subject 3 via the packing map.

    Subject 1 is the reference grid: its ``id`` equals the original sentence
    index for every row. Subject 3's ``id`` is packed. ``SentLen`` must match
    on the reconstructed pairing because sentence length is a stimulus
    property, not a reader property.
    """
    s1 = subject1.set_index("id", drop=False)
    rows = []
    for packed_id, row in subject3.set_index("id", drop=False).iterrows():
        original = unpack_subject3_id(int(packed_id))
        ref = s1.loc[original]
        rows.append(
            {
                "packed_id": int(packed_id),
                "original_id": original,
                "s3_SentLen": float(row["SentLen"]),
                "s1_SentLen": float(ref["SentLen"]),
                "sentlen_match": float(row["SentLen"]) == float(ref["SentLen"]),
                "same_row_would_match": (
                    float(row["SentLen"]) == float(s1.loc[int(packed_id), "SentLen"])
                    if int(packed_id) in s1.index
                    else False
                ),
            }
        )
    return pd.DataFrame(rows)


def word_break_row(subject1_word: pd.DataFrame) -> int:
    """First word-table row that belongs to original sentence 150.

    Word-level CSVs use ``Sent_ID`` like ``150_NR``. After packing, subject 3
    still labels that row ``150_NR``, but the *tokens* are original sentence
    250. The positional average used by ``word/get_average.py`` therefore
    mixes readers starting at this row index.
    """
    sid = subject1_word["Sent_ID"].astype(str).str.split("_").str[0].astype(int)
    return int((sid < PACK_SHIFT_AFTER).sum())
