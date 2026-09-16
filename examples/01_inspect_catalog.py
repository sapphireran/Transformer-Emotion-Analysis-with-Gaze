#!/usr/bin/env python3
"""Print the personal dataset catalog and confirm every file opens."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from gaze_emotion_examples.catalog import DATASETS
from gaze_emotion_examples.io import load_dataset


def main() -> None:
    print(f"{'key':<22} {'kind':<10} {'rows':>7}  path")
    print("-" * 88)
    for spec in DATASETS:
        exists = spec.path.exists()
        if not exists:
            print(f"{spec.key:<22} {spec.kind:<10} {'MISSING':>7}  {spec.relative_path}")
            continue
        frame = load_dataset(spec)
        print(f"{spec.key:<22} {spec.kind:<10} {len(frame):7d}  {spec.relative_path}")
        print(f"  {spec.title}")


if __name__ == "__main__":
    main()
