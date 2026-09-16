"""Train / valid / test inventory and leakage checks."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def disjoint_id_sets(*frames: pd.DataFrame, id_col: str = "sentence_id") -> bool:
    ids = [set(df[id_col].tolist()) for df in frames]
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if a & b:
                return False
    return True


def overlapping_ids(
    *frames: pd.DataFrame, id_col: str = "sentence_id"
) -> dict[tuple[int, int], set]:
    ids = [set(df[id_col].tolist()) for df in frames]
    out: dict[tuple[int, int], set] = {}
    for i, a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            ov = a & ids[j]
            if ov:
                out[(i, j)] = ov
    return out


def covers_exactly(
    parts: Sequence[pd.DataFrame], universe: pd.DataFrame, id_col: str = "sentence_id"
) -> bool:
    union = set()
    for df in parts:
        union |= set(df[id_col].tolist())
    return union == set(universe[id_col].tolist())


def split_inventory(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    combined: pd.DataFrame,
    id_col: str = "sentence_id",
) -> dict:
    parts = (train, valid, test)
    return {
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "n_test": int(len(test)),
        "n_combined": int(len(combined)),
        "disjoint": disjoint_id_sets(*parts, id_col=id_col),
        "covers_combined": covers_exactly(parts, combined, id_col=id_col),
        "overlaps": {
            f"{a}-{b}": sorted(v)[:10]
            for (a, b), v in overlapping_ids(*parts, id_col=id_col).items()
        },
        "train_labels": train["sentiment_label"].value_counts().sort_index().to_dict()
        if "sentiment_label" in train.columns
        else {},
        "valid_labels": valid["sentiment_label"].value_counts().sort_index().to_dict()
        if "sentiment_label" in valid.columns
        else {},
        "test_labels": test["sentiment_label"].value_counts().sort_index().to_dict()
        if "sentiment_label" in test.columns
        else {},
    }
