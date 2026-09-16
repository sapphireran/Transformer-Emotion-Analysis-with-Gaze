"""Reader-3 compaction and the published positional average.

``utils_ZuCo.DataTransformer`` (task 1, subject 2) skips MATLAB sentence
indices 150–249 and 399, then writes ``ZuCo_et_csv_data/3_SR.csv`` with a
fresh 0..298 ``id``. ``get_average_sentence_level.py`` concatenates the
twelve CSVs and groups by *row index*, treating 0 as missing. The first
150 sentences stay aligned. After that, reader 3's original sentence
``k + 100`` is folded into published sentence ``k``.

Word-level files have the same trap: the first 2594 tokens (sentences
0–149) match across readers; every later row averages different words.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .csvio import float_col, read_dicts
from .paths import repo_root

N_SENTENCES = 400
N_READERS = 12
READER3_INDEX = 2  # 0-based; file is 3_SR.csv
READER3_COMPACT_ROWS = 299
CLEAN_SENTENCE_END = 150  # original ids 0..149 are 12-reader aligned
WORD_ALIGN_ROWS = 2594  # tokens belonging to sentences 0..149
DROPPED_ORIGINAL_IDS = frozenset(range(150, 250)) | {399}

# Measured on this clone (0→NaN positional average vs remapped 0→NaN average).
NFIX_CONTAMINATED = 246
SENTLEN_CONTAMINATED = 147
WORST_NFIX_ID = 239
WORD_MISMATCH_IN_OVERLAP = 2674  # reader 1 vs reader 3, first 5293 index rows


def compact_to_original(compact_id: int) -> int:
    """Map a ``3_SR.csv`` id back to the MATLAB sentence index."""
    if compact_id < 0 or compact_id > 298:
        raise ValueError(f"compact id {compact_id} is outside 0..298")
    if compact_id <= 149:
        return compact_id
    return compact_id + 100  # 150 → 250, 298 → 398


def original_to_compact(original_id: int) -> int | None:
    """Return reader 3's compact id, or None if that sentence was dropped."""
    if original_id < 0 or original_id > 399:
        raise ValueError(f"original id {original_id} is outside 0..399")
    if original_id <= 149:
        return original_id
    if 250 <= original_id <= 398:
        return original_id - 100
    return None


def load_subject_tables(root=None) -> list[list[dict[str, str]]]:
    root = root or repo_root()
    tables = []
    for i in range(1, 13):
        _, rows = read_dicts(root / f"ZuCo_et_csv_data/{i}_SR.csv")
        tables.append(rows)
    return tables


def _values_at_index(tables: list[list[dict[str, str]]], idx: int, col: str) -> list[float]:
    vals = []
    for rows in tables:
        if idx < len(rows):
            vals.append(float(rows[idx][col]))
    return vals


def _values_realigned(tables: list[list[dict[str, str]]], orig: int, col: str) -> list[float]:
    vals = []
    for si, rows in enumerate(tables):
        if si == READER3_INDEX:
            compact = original_to_compact(orig)
            if compact is None:
                continue
            vals.append(float(rows[compact][col]))
        else:
            vals.append(float(rows[orig][col]))
    return vals


def nanmean_skip_zero(values: list[float]) -> float:
    kept = [v for v in values if v != 0.0]
    if not kept:
        return float("nan")
    return float(np.mean(kept))


def positional_average(tables: list[list[dict[str, str]]], col: str) -> np.ndarray:
    """Reproduce ``get_average_sentence_level.py`` for one column (0 → skip)."""
    return np.array(
        [nanmean_skip_zero(_values_at_index(tables, i, col)) for i in range(N_SENTENCES)],
        dtype=np.float64,
    )


def remapped_average(tables: list[list[dict[str, str]]], col: str) -> np.ndarray:
    """Same 0→skip mean, but reader 3 is looked up by original sentence id."""
    return np.array(
        [nanmean_skip_zero(_values_realigned(tables, i, col)) for i in range(N_SENTENCES)],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class Contamination:
    column: str
    n_changed: int
    mean_abs: float
    max_abs: float
    worst_id: int
    published: np.ndarray
    remapped: np.ndarray
    delta: np.ndarray


def contamination_report(
    tables: list[list[dict[str, str]]],
    published: np.ndarray,
    col: str,
) -> Contamination:
    remapped = remapped_average(tables, col)
    delta = remapped - published
    absd = np.abs(delta)
    # Treat tiny float noise as equal.
    changed = absd > 1e-9
    worst = int(np.nanargmax(absd))
    return Contamination(
        column=col,
        n_changed=int(np.sum(changed)),
        mean_abs=float(np.nanmean(absd)),
        max_abs=float(np.nanmax(absd)),
        worst_id=worst,
        published=published,
        remapped=remapped,
        delta=delta,
    )


def published_column(root=None, col: str = "nFixations") -> np.ndarray:
    root = root or repo_root()
    _, rows = read_dicts(root / "ZuCo_et_csv_data/average_data.csv")
    return float_col(rows, col)


def word_align_row_count(root=None) -> int:
    """First index where reader 1 and reader 3 store different words."""
    root = root or repo_root()
    _, w1 = read_dicts(root / "ZuCo_et_csv_data/word/1_SR.csv")
    _, w3 = read_dicts(root / "ZuCo_et_csv_data/word/3_SR.csv")
    for i, (a, b) in enumerate(zip(w1, w3)):
        if a["Word"] != b["Word"] or a["Sent_ID"] != b["Sent_ID"]:
            return i
    return min(len(w1), len(w3))


def word_mismatch_count(root=None) -> int:
    root = root or repo_root()
    _, w1 = read_dicts(root / "ZuCo_et_csv_data/word/1_SR.csv")
    _, w3 = read_dicts(root / "ZuCo_et_csv_data/word/3_SR.csv")
    return sum(1 for a, b in zip(w1, w3) if a["Word"] != b["Word"])
