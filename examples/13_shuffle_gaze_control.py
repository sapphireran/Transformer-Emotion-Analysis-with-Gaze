#!/usr/bin/env python3
"""Shuffle labels and ask whether observed gaze ANOVA F is unusual.

For each feature, compare the real F-statistic to the F's you get after
permuting sentiment_label. perm_p is the fraction of shuffles with F at
least as large as the observed one (plus a floor of 0 if none beat it).

This is the cheap personal check before believing example 03's
p-values, which assume independent Gaussian class-conditional features
— a bad description of predicted gaze.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.baselines import permute_anova
from examples.lib.gaze import FULL_SST_GAZE, ZUCO_FUSION_GAZE
from examples.lib.loading import documented_datasets, load_dataset, load_zuco_combined
from examples.lib.paths import resolve_root
from examples.lib.reporting import banner, print_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--n-perm", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    root = resolve_root(args.root)

    sst = load_dataset(
        next(s for s in documented_datasets() if s.key == "sst_combined"), root=root
    )
    zuco = load_zuco_combined(root=root)

    banner(f"Full SST predicted gaze, {args.n_perm} label shuffles")
    sst_table = permute_anova(sst, FULL_SST_GAZE, n_perm=args.n_perm, seed=args.seed)
    print_frame(sst_table)

    banner(f"ZuCo 400 measured gaze, {args.n_perm} label shuffles")
    zuco_table = permute_anova(zuco, ZUCO_FUSION_GAZE, n_perm=args.n_perm, seed=args.seed)
    print_frame(zuco_table)
    print(
        "\nperm_p near 0: the real class split is unusual under shuffled "
        "labels. perm_p near 0.5: the observed F is typical of noise. "
        "On n=400, even a 'significant' sklearn p-value can have a "
        "mediocre permutation p — believe the permutation column."
    )

    if args.output_dir:
        out = Path(args.output_dir)
        if not out.is_absolute():
            out = root / out
        out.mkdir(parents=True, exist_ok=True)
        sst_path = out / "permute_anova_full_sst.csv"
        zuco_path = out / "permute_anova_zuco.csv"
        sst_table.to_csv(sst_path, index=False)
        zuco_table.to_csv(zuco_path, index=False)
        note = {
            "n_perm": args.n_perm,
            "seed": args.seed,
            "sst_min_perm_p": float(sst_table["perm_p"].min()),
            "zuco_min_perm_p": float(zuco_table["perm_p"].min()),
        }
        note_path = out / "permute_anova.json"
        note_path.write_text(json.dumps(note, indent=2), encoding="utf-8")
        print(f"\nwrote {sst_path}")
        print(f"wrote {zuco_path}")
        print(f"wrote {note_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
