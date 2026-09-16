#!/usr/bin/env python3
"""Write docs/generated/dataset-report.md from the checked-in CSVs."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from gaze_emotion_examples.reports import write_dataset_report


def main() -> None:
    path = write_dataset_report()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
