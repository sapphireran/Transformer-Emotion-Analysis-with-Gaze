#!/usr/bin/env python3
"""Per-class gaze means, correlations, and ANOVA vs sentiment label."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.gaze import (
    FULL_SST_GAZE,
    ZUCO_ALL_GAZE,
    feature_correlation,
    feature_label_anova,
    per_class_means,
)
from examples.lib.loading import documented_datasets, load_dataset, load_zuco_combined
from examples.lib.paths import resolve_root
from examples.lib.reporting import banner, print_frame


def _block(title: str, df, features) -> None:
    banner(f"{title}: per-class means")
    print_frame(per_class_means(df, features))
    banner(f"{title}: Pearson correlation")
    print_frame(feature_correlation(df, features))
    banner(f"{title}: ANOVA F vs sentiment_label")
    print_frame(feature_label_anova(df, features))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    root = resolve_root(args.root)

    sst = load_dataset(
        next(s for s in documented_datasets() if s.key == "sst_combined"), root=root
    )
    zuco = load_zuco_combined(root=root)

    _block("Full SST (predicted / aligned gaze, z-scored)", sst, FULL_SST_GAZE)
    _block("ZuCo 400 (measured gaze, standardized)", zuco, ZUCO_ALL_GAZE)

    print(
        "\nReading the ANOVA table: a large F means the class-conditional "
        "means of that feature differ. On predicted full-SST gaze the "
        "effects are often tiny (the features were not collected as "
        "emotion labels). On ZuCo, omissionRate and the late measures "
        "(TRT, GPT) are the ones worth staring at first."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
