#!/usr/bin/env python3
"""Audit the stored 80/10/10 splits for ID leakage and label drift."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from gaze_emotion_examples.io import load_dataset
from gaze_emotion_examples.splits import audit_split, audit_to_markdown


def main() -> None:
    zuco = audit_split(
        "ZuCo ∩ SST",
        load_dataset("zuco_sst_standard"),
        load_dataset("zuco_sst_train"),
        load_dataset("zuco_sst_valid"),
        load_dataset("zuco_sst_test"),
        id_column="sentence_id",
        label_column="sentiment_label",
    )
    sst = audit_split(
        "Full SST",
        load_dataset("sst_full"),
        load_dataset("sst_train"),
        load_dataset("sst_valid"),
        load_dataset("sst_test"),
        id_column="sentence_id",
        label_column="sentiment_label",
    )
    print(audit_to_markdown(zuco))
    print()
    print(audit_to_markdown(sst))
    if not (zuco.ok and sst.ok):
        raise SystemExit("split audit failed")
    print("\nboth split families are disjoint and cover their parent tables")


if __name__ == "__main__":
    main()
