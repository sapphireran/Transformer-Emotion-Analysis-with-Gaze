"""Holdout-split sanity checks (leakage, class drift, id overlap)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SplitReport:
    n_train: int
    n_valid: int
    n_test: int
    id_column: str
    overlapping_ids: dict[str, list]
    missing_from_union: int
    extra_in_union: int
    label_fractions: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    @property
    def is_disjoint(self) -> bool:
        return all(len(v) == 0 for v in self.overlapping_ids.values())


def _ids(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise KeyError(f"split check needs {column}")
    return df[column]


def check_splits(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    id_column: str = "sentence_id",
    label_column: str = "sentiment_label",
    expected_union: pd.DataFrame | None = None,
) -> SplitReport:
    """Compare three holdout frames.

    Parameters
    ----------
    expected_union:
        Optional combined table. When given, the check reports how many ids in
        the union are missing from the three splits and vice versa.
    """
    t, v, te = _ids(train, id_column), _ids(valid, id_column), _ids(test, id_column)
    overlaps = {
        "train∩valid": sorted(set(t) & set(v)),
        "train∩test": sorted(set(t) & set(te)),
        "valid∩test": sorted(set(v) & set(te)),
    }
    notes: list[str] = []
    missing = extra = 0
    if expected_union is not None:
        union_ids = set(_ids(expected_union, id_column))
        split_ids = set(t) | set(v) | set(te)
        missing = len(union_ids - split_ids)
        extra = len(split_ids - union_ids)
        if missing:
            notes.append(f"{missing} ids from the combined table never appear in a split")
        if extra:
            notes.append(f"{extra} split ids are not in the combined table")
        if train.duplicated(id_column).any() or valid.duplicated(id_column).any() or test.duplicated(id_column).any():
            notes.append("duplicate ids inside at least one split frame")

    def _frac(df: pd.DataFrame, name: str) -> pd.DataFrame:
        vc = df[label_column].value_counts(normalize=True).sort_index()
        out = vc.rename(name).to_frame()
        return out

    label_fractions = pd.concat(
        [_frac(train, "train"), _frac(valid, "valid"), _frac(test, "test")],
        axis=1,
    ).fillna(0.0)
    if expected_union is not None and label_column in expected_union.columns:
        label_fractions = label_fractions.join(
            expected_union[label_column].value_counts(normalize=True).rename("combined"),
            how="outer",
        ).fillna(0.0)

    if not all(len(v) == 0 for v in overlaps.values()):
        notes.append("id overlap across splits — leakage")

    return SplitReport(
        n_train=len(train),
        n_valid=len(valid),
        n_test=len(test),
        id_column=id_column,
        overlapping_ids=overlaps,
        missing_from_union=missing,
        extra_in_union=extra,
        label_fractions=label_fractions,
        notes=notes,
    )


def max_class_drift(report: SplitReport, versus: str = "combined") -> float:
    """Largest absolute fraction gap between a split and ``versus``."""
    frac = report.label_fractions
    if versus not in frac.columns:
        versus = "train"
    drift = 0.0
    for col in ("train", "valid", "test"):
        if col in frac.columns and col != versus:
            drift = max(drift, float((frac[col] - frac[versus]).abs().max()))
    return drift
