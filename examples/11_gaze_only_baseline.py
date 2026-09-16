#!/usr/bin/env python3
"""Gaze-only logistic regression: does ET predict polarity without BERT?

Fits a scaled multinomial logistic regression on the same five features
the fusion head sees. No transformer, no GPU.

- ZuCo 400: 5-fold stratified CV (matches model_ZuCo_SST.py's protocol).
- Full SST: train on the 80% split, score valid and test.

Also reports a *length-only* model so a length confound is visible.
A gaze model that cannot beat majority *or* length is not carrying
polarity information worth fusing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.lib.baselines import holdout_scores, stratified_cv_scores
from examples.lib.gaze import FULL_SST_GAZE, ZUCO_ALL_GAZE, ZUCO_FUSION_GAZE
from examples.lib.loading import load_full_sst_splits, load_zuco_combined
from examples.lib.paths import resolve_root
from examples.lib.reporting import banner, print_frame

import pandas as pd


def _row(name: str, scores: dict) -> dict:
    keep = {
        "setup": name,
        "accuracy": scores.get("accuracy"),
        "f1_weighted": scores.get("f1_weighted"),
        "majority_accuracy": scores.get("majority_accuracy"),
        "accuracy_minus_majority": scores.get("accuracy_minus_majority"),
    }
    if "n" in scores:
        keep["n"] = scores["n"]
    if "n_test" in scores:
        keep["n_test"] = scores["n_test"]
    return keep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = resolve_root(args.root)

    zuco = load_zuco_combined(root=root)
    zuco = zuco.copy()
    zuco["n_tokens"] = zuco["sentence"].astype(str).str.split().str.len()

    banner("ZuCo 400 — 5-fold stratified CV, gaze-only logreg")
    zuco_rows = [
        _row("fusion5 (nFixations,FFD,GPT,TRT,GD)", stratified_cv_scores(zuco[list(ZUCO_FUSION_GAZE)], zuco["sentiment_label"], seed=args.seed)),
        _row("all ZuCo ET columns", stratified_cv_scores(zuco[list(ZUCO_ALL_GAZE)], zuco["sentiment_label"], seed=args.seed)),
        _row("length only (whitespace tokens)", stratified_cv_scores(zuco[["n_tokens"]], zuco["sentiment_label"], seed=args.seed)),
    ]
    zuco_table = pd.DataFrame(zuco_rows)
    print_frame(zuco_table)
    print(
        "\nIf fusion5 cannot beat majority on 400 measured-gaze sentences, "
        "concatenating those five numbers to RoBERTa is hoping the encoder "
        "will use a weak side channel. SFD/omissionRate are in 'all ZuCo ET' "
        "but not in the training scripts."
    )

    splits = load_full_sst_splits(root=root)
    for name, df in splits.items():
        df["n_tokens"] = df["sentence"].astype(str).str.split().str.len()
        splits[name] = df
    banner("Full SST — train logreg on train split, score valid/test")
    sst_rows = []
    for split_name in ("valid", "test"):
        sst_rows.append(
            _row(
                f"fusion5 → {split_name}",
                holdout_scores(
                    splits["train"][list(FULL_SST_GAZE)],
                    splits["train"]["sentiment_label"],
                    splits[split_name][list(FULL_SST_GAZE)],
                    splits[split_name]["sentiment_label"],
                    seed=args.seed,
                ),
            )
        )
        sst_rows.append(
            _row(
                f"length only → {split_name}",
                holdout_scores(
                    splits["train"][["n_tokens"]],
                    splits["train"]["sentiment_label"],
                    splits[split_name][["n_tokens"]],
                    splits[split_name]["sentiment_label"],
                    seed=args.seed,
                ),
            )
        )
    sst_table = pd.DataFrame(sst_rows)
    print_frame(sst_table)
    print(
        "\nFull-SST gaze columns are predicted/aligned, not twelve new ZuCo "
        "subjects. A gaze-only model that beats majority here can mean the "
        "predictor leaked polarity into the features — treat that as a "
        "warning, not as evidence that humans read positive reviews faster."
    )

    payload = {
        "zuco": zuco_rows,
        "full_sst": sst_rows,
        "seed": args.seed,
    }
    if args.output_dir:
        out = Path(args.output_dir)
        if not out.is_absolute():
            out = root / out
        out.mkdir(parents=True, exist_ok=True)
        table_path = out / "gaze_only_baseline.csv"
        json_path = out / "gaze_only_baseline.json"
        pd.concat(
            [zuco_table.assign(track="zuco"), sst_table.assign(track="full_sst")],
            ignore_index=True,
        ).to_csv(table_path, index=False)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {table_path}")
        print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
