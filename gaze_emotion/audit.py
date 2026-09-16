"""Dataset audits used by the examples: balance, leakage, subject coverage."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import ZUCO_SUBJECT_COUNT, ZUCO_TASK1_EXPECTED_SENTENCES
from .datasets import (
    REPO_ROOT,
    load_csv_rows,
    sentence_ids,
    split_overlap,
    summarize_rows,
)
from .scaling import average_subject_tables


@dataclass(frozen=True)
class SplitAudit:
    train_rows: int
    valid_rows: int
    test_rows: int
    train_valid_overlap: int
    train_test_overlap: int
    valid_test_overlap: int
    train_labels: dict[str, int]
    valid_labels: dict[str, int]
    test_labels: dict[str, int]

    @property
    def has_leakage(self) -> bool:
        return bool(self.train_valid_overlap or self.train_test_overlap or self.valid_test_overlap)


def audit_splits(train_path: str, valid_path: str, test_path: str) -> SplitAudit:
    train = load_csv_rows(train_path)
    valid = load_csv_rows(valid_path)
    test = load_csv_rows(test_path)
    return SplitAudit(
        train_rows=len(train),
        valid_rows=len(valid),
        test_rows=len(test),
        train_valid_overlap=len(split_overlap(train, valid)),
        train_test_overlap=len(split_overlap(train, test)),
        valid_test_overlap=len(split_overlap(valid, test)),
        train_labels=summarize_rows(train).label_counts,
        valid_labels=summarize_rows(valid).label_counts,
        test_labels=summarize_rows(test).label_counts,
    )


def format_split_audit(name: str, audit: SplitAudit) -> str:
    lines = [
        f"## {name}",
        f"sizes: train={audit.train_rows} valid={audit.valid_rows} test={audit.test_rows}",
        f"sentence_id overlap: train∩valid={audit.train_valid_overlap} "
        f"train∩test={audit.train_test_overlap} valid∩test={audit.valid_test_overlap}",
        f"leakage detected: {'yes' if audit.has_leakage else 'no'}",
        f"train labels: {audit.train_labels}",
        f"valid labels: {audit.valid_labels}",
        f"test labels:  {audit.test_labels}",
    ]
    return "\n".join(lines)


def subject_sentence_counts(directory: str | Path = "ZuCo_et_csv_data") -> dict[str, int]:
    folder = Path(directory)
    if not folder.is_absolute():
        folder = REPO_ROOT / folder
    counts: dict[str, int] = {}
    for idx in range(1, ZUCO_SUBJECT_COUNT + 1):
        path = folder / f"{idx}_SR.csv"
        if path.exists():
            counts[path.name] = len(load_csv_rows(path))
    return counts


def subject_coverage_notes(counts: dict[str, int]) -> list[str]:
    notes = []
    for name, n in counts.items():
        if n != ZUCO_TASK1_EXPECTED_SENTENCES:
            notes.append(
                f"{name} has {n} sentence rows instead of "
                f"{ZUCO_TASK1_EXPECTED_SENTENCES}. This matches the skip rules "
                "in utils_ZuCo.DataTransformer for incomplete ZuCo recordings."
            )
    if not notes:
        notes.append("All subject files have the expected 400 sentence rows.")
    return notes


def reconstruct_sentence_average(directory: str | Path = "ZuCo_et_csv_data") -> np.ndarray:
    """Rebuild the subject-mean table used before sklearn scaling."""
    folder = Path(directory)
    if not folder.is_absolute():
        folder = REPO_ROOT / folder
    tables = []
    feature_cols = [
        "SentLen",
        "omissionRate",
        "nFixations",
        "meanPupilSize",
        "GD",
        "TRT",
        "FFD",
        "SFD",
        "GPT",
    ]
    for idx in range(1, ZUCO_SUBJECT_COUNT + 1):
        rows = load_csv_rows(folder / f"{idx}_SR.csv")
        matrix = np.array(
            [[float(row[col]) if row[col] not in ("", None) else np.nan for col in feature_cols] for row in rows],
            dtype=float,
        )
        tables.append(matrix)
    # Subject 3 is shorter; align by taking the first min_rows shared ids is
    # not valid here because the conversion already dropped bad trials. The
    # original script mean-stacks on index, so we pad with NaN.
    max_rows = max(table.shape[0] for table in tables)
    padded = []
    for table in tables:
        if table.shape[0] == max_rows:
            padded.append(table)
            continue
        pad = np.full((max_rows - table.shape[0], table.shape[1]), np.nan)
        padded.append(np.vstack([table, pad]))
    return average_subject_tables(padded)


def label_entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    probs = np.array([c / total for c in counts.values()], dtype=float)
    return float(-np.sum(probs * np.log2(np.clip(probs, 1e-12, 1.0))))


def majority_baseline(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return max(counts.values()) / total


def count_zero_gaze_rows(path: str, columns: tuple[str, ...]) -> int:
    rows = load_csv_rows(path)
    zeros = 0
    for row in rows:
        values = [float(row[col]) for col in columns]
        if all(abs(v) < 1e-15 for v in values):
            zeros += 1
    return zeros


def tally_words_per_sentence(path: str) -> Counter[int]:
    rows = load_csv_rows(path)
    if not rows or "sentence_id" not in rows[0]:
        raise KeyError("Expected a word-level file with sentence_id")
    per_sentence: Counter[str] = Counter(row["sentence_id"] for row in rows)
    return Counter(per_sentence.values())
