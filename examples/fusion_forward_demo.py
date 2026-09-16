#!/usr/bin/env python3
"""Train tiny softmax heads that rehearse the late-fusion layout.

Replaces BERT/RoBERTa with a 32-d hashed bag-of-words so the concat +
linear arithmetic in docs/architecture.md can run on CPU in a few seconds.

Compares:
  1. majority class
  2. gaze-only (5 z-scored ZuCo features)
  3. text-only (hash embedding)
  4. fused (hash ⊕ Linear(5→16))
  5. fused with row-shuffled gaze (ablation)

This is a teaching baseline, not a RoBERTa score.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.lib import cli, features, fusion, io_csv, metrics, paths, report, schema, stats


TEXT_DIM = 32
EPOCHS = 45
HOLDOUT = 0.25
SEED = 42


def load_zuco():
    rows = io_csv.read_rows(paths.table_path("zuco_standard"))
    labels = features.labels_of(rows)
    texts = [row["sentence"] for row in rows]
    gaze = stats.zscore_columns(features.matrix_for(rows, "zuco5"))
    text = fusion.embed_texts(texts, dim=TEXT_DIM)
    return rows, labels, text, gaze


def evaluate_constant(name: str, y_true, y_pred) -> fusion.TrainResult:
    rec = metrics.report(y_true, y_pred)
    return fusion.TrainResult(
        name=name,
        train_acc=float("nan"),
        holdout_acc=float(rec["accuracy"]),
        holdout_macro_f1=float(rec["macro"]["f1"]),
        holdout_report=rec,
        losses=[],
    )


def main() -> int:
    args = cli.parser("CPU late-fusion demo on ZuCo.").parse_args()
    rows, labels, text, gaze = load_zuco()
    indices = list(range(len(rows)))
    train_idx, test_idx = fusion.stratified_split(
        indices, labels, test_ratio=HOLDOUT, seed=SEED
    )
    y_test = [labels[i] for i in test_idx]
    y_train = [labels[i] for i in train_idx]

    majority_preds = fusion.majority_predict(y_train, len(y_test))
    majority = evaluate_constant("majority", y_test, majority_preds)

    gaze_only = fusion.train_softmax_head(
        "gaze-only",
        gaze,
        labels,
        train_idx,
        test_idx,
        epochs=EPOCHS,
        lr=0.18,
        seed=1,
    )
    text_only = fusion.train_softmax_head(
        "text-only (hash)",
        text,
        labels,
        train_idx,
        test_idx,
        epochs=EPOCHS,
        lr=0.22,
        seed=2,
    )
    fused = fusion.train_fusion_head(
        "fused (hash ⊕ gaze)",
        text,
        gaze,
        labels,
        train_idx,
        test_idx,
        epochs=EPOCHS,
        lr=0.14,
        gaze_out=16,
        seed=3,
    )
    shuffled = fusion.shuffle_rows(gaze, seed=99)
    fused_shuffled = fusion.train_fusion_head(
        "fused + shuffled gaze",
        text,
        shuffled,
        labels,
        train_idx,
        test_idx,
        epochs=EPOCHS,
        lr=0.14,
        gaze_out=16,
        seed=3,
    )

    results = [majority, gaze_only, text_only, fused, fused_shuffled]
    table = report.ascii_table(
        ("model", "train acc", "holdout acc", "holdout macro F1", "final loss"),
        [
            (
                r.name,
                r.train_acc if r.losses else "—",
                r.holdout_acc,
                r.holdout_macro_f1,
                r.losses[-1] if r.losses else "—",
            )
            for r in results
        ],
    )
    conf = report.format_confusion(
        fused.holdout_report["confusion"],
        [schema.LABEL_NAMES[i] for i in (0, 1, 2)],
    )
    notes = [
        f"{len(rows)} ZuCo rows, stratified holdout {len(test_idx)} "
        f"({HOLDOUT:.0%}), seed={SEED}, epochs={EPOCHS}.",
        "Text vector is a 32-d hashed bag of words, not RoBERTa.",
        "Gaze input is the z-scored 5-d set (nFixations, FFD, GPT, TRT, GD).",
        "If shuffled-gaze ≈ fused, the gaze branch is not carrying signal "
        "in this tiny head. If it is worse, the real gaze values were used.",
        "Do not paste these accuracies next to GPU transformer runs.",
    ]
    body = report.join_sections(
        [
            "# Late-fusion CPU demo (ZuCo)",
            table,
            "Confusion matrix for fused (hash ⊕ gaze) on the holdout:",
            conf,
            report.bullet(notes),
        ]
    )
    print(body)
    written = cli.maybe_write(args, "fusion_forward_demo.md", body)
    if written:
        print(f"\nwrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
