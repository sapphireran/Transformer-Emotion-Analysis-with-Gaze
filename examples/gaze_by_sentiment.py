#!/usr/bin/env python3
"""Class-conditional gaze means on ZuCo gold and full-SST transferred features."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from csvutil import float_column, mean, pstdev, read_dicts
from paths import (
    LABEL_NAMES,
    SST_COMBINED,
    SST_GAZE_COLS,
    ZUCO_ALL_GAZE_COLS,
    ZUCO_COMBINED_STD,
    ZUCO_MODEL_GAZE_COLS,
)


def grouped_stats(path, feature_names) -> dict[str, dict[str, dict[str, float]]]:
    _, rows = read_dicts(path)
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        label = row["sentiment_label"]
        for name in feature_names:
            buckets[label][name].append(float(row[name]))
    stats = {}
    for label, features in buckets.items():
        stats[label] = {
            name: {"n": len(values), "mean": mean(values), "std": pstdev(values)}
            for name, values in features.items()
        }
    return dict(sorted(stats.items()))


def format_block(title: str, stats: dict, feature_names) -> str:
    lines = [title]
    for label, features in stats.items():
        n = features[feature_names[0]]["n"]
        lines.append(f"  {label} {LABEL_NAMES[label]} n={n}")
        for name in feature_names:
            item = features[name]
            lines.append(f"    {name:16s} mean={item['mean']:+.4f}  std={item['std']:.4f}")
    return "\n".join(lines)


def pairwise_mean_gap(stats: dict, feature: str, left: str, right: str) -> float:
    return stats[left][feature]["mean"] - stats[right][feature]["mean"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    zuco = grouped_stats(ZUCO_COMBINED_STD, ZUCO_ALL_GAZE_COLS)
    sst = grouped_stats(SST_COMBINED, SST_GAZE_COLS)

    if not args.quiet:
        print(format_block("ZuCo combined (z-scored real gaze)", zuco, ZUCO_ALL_GAZE_COLS))
        print()
        print(format_block("Full SST (transferred / predicted gaze)", sst, SST_GAZE_COLS))
        print()
        print("Gaps that show up in the docs:")
        print(
            "  ZuCo nFixations  neutral - negative = "
            f"{pairwise_mean_gap(zuco, 'nFixations', '1', '0'):+.4f}"
        )
        print(
            "  ZuCo GD          neutral - negative = "
            f"{pairwise_mean_gap(zuco, 'GD', '1', '0'):+.4f}"
        )
        print(
            "  SST nFix         positive - negative = "
            f"{pairwise_mean_gap(sst, 'nFix', '2', '0'):+.4f}"
        )
        print(
            "Model gaze channels:",
            ", ".join(ZUCO_MODEL_GAZE_COLS),
            "| unused: omissionRate, meanPupilSize, SFD",
        )

    # Cheap sanity: every class exists, std is finite and not identically zero
    # on the features the model uses.
    for label in ("0", "1", "2"):
        if label not in zuco or label not in sst:
            print(f"missing label {label}")
            return 1
        if zuco[label]["nFixations"]["std"] <= 0:
            print("ZuCo nFixations has zero variance")
            return 1

    # The z-scored table should be globally centered even if class means drift.
    _, zuco_rows = read_dicts(ZUCO_COMBINED_STD)
    global_nfix = mean(float_column(zuco_rows, "nFixations"))
    if abs(global_nfix) > 1e-6:
        print(f"ZuCo nFixations is not globally centered: {global_nfix}")
        return 1

    print("gaze-by-sentiment OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
