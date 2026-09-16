"""Sanity checks for the stored train/valid/test CSVs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitAudit:
    name: str
    train_rows: int
    valid_rows: int
    test_rows: int
    union_rows: int
    disjoint: bool
    covers_parent: bool
    extra_ids: tuple
    missing_ids: tuple
    label_drift: dict[str, dict[int, float]]

    @property
    def ok(self) -> bool:
        return self.disjoint and self.covers_parent and not self.extra_ids


def _id_set(frame: pd.DataFrame, id_column: str) -> set:
    return set(frame[id_column].tolist())


def _label_share(frame: pd.DataFrame, label_column: str | None) -> dict[int, float]:
    if not label_column or label_column not in frame.columns or frame.empty:
        return {}
    shares = frame[label_column].value_counts(normalize=True)
    return {int(code): float(share) for code, share in shares.items()}


def audit_split(
    name: str,
    parent: pd.DataFrame,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    id_column: str,
    label_column: str | None = None,
) -> SplitAudit:
    """Check disjoint IDs, coverage of the parent table, and label mix drift."""
    train_ids = _id_set(train, id_column)
    valid_ids = _id_set(valid, id_column)
    test_ids = _id_set(test, id_column)
    parent_ids = _id_set(parent, id_column)
    union = train_ids | valid_ids | test_ids
    disjoint = (
        train_ids.isdisjoint(valid_ids)
        and train_ids.isdisjoint(test_ids)
        and valid_ids.isdisjoint(test_ids)
    )
    extra = tuple(sorted(union - parent_ids, key=lambda value: str(value)))
    missing = tuple(sorted(parent_ids - union, key=lambda value: str(value)))
    drift = {
        "parent": _label_share(parent, label_column),
        "train": _label_share(train, label_column),
        "valid": _label_share(valid, label_column),
        "test": _label_share(test, label_column),
    }
    return SplitAudit(
        name=name,
        train_rows=len(train),
        valid_rows=len(valid),
        test_rows=len(test),
        union_rows=len(union),
        disjoint=disjoint,
        covers_parent=not missing,
        extra_ids=extra,
        missing_ids=missing,
        label_drift=drift,
    )


def audit_to_markdown(audit: SplitAudit) -> str:
    """Render one split audit as markdown."""
    status = "ok" if audit.ok else "needs attention"
    lines = [
        f"### {audit.name} ({status})",
        "",
        f"- train/valid/test rows: {audit.train_rows} / {audit.valid_rows} / {audit.test_rows}",
        f"- unique union IDs: {audit.union_rows}",
        f"- disjoint splits: {audit.disjoint}",
        f"- covers parent table: {audit.covers_parent}",
    ]
    if audit.extra_ids:
        lines.append(f"- extra IDs not in parent: {list(audit.extra_ids)[:12]}")
    if audit.missing_ids:
        lines.append(f"- parent IDs missing from splits: {list(audit.missing_ids)[:12]}")
    parent = audit.label_drift.get("parent", {})
    if parent:
        lines.append("- label mix (negative / neutral / positive):")
        for split_name in ("parent", "train", "valid", "test"):
            shares = audit.label_drift.get(split_name, {})
            pretty = ", ".join(
                f"{shares.get(code, 0.0) * 100:.1f}%" for code in (0, 1, 2)
            )
            lines.append(f"  - {split_name}: {pretty}")
    return "\n".join(lines)
