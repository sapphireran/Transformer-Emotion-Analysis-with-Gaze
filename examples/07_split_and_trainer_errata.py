#!/usr/bin/env python3
"""Audit the unused 80/10/10 files and the full-SST last-batch test loop."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gazebook.csvio import read_dicts
from gazebook.paths import repo_root
from gazebook.reports import md_table, write_text
from gazebook.schema import FULL_SST_BATCH, FULL_SST_LAST_BATCH, FULL_SST_TEST_ROWS


def _ids(path, key="sentence_id"):
    _, rows = read_dicts(path)
    return [int(r[key]) for r in rows], [int(r["sentiment_label"]) for r in rows]


def _mix(labels):
    c = Counter(labels)
    return c[0], c[1], c[2]


def main() -> int:
    root = repo_root()
    z_tr, z_tr_y = _ids(root / "ZuCo_SST_data/train.csv")
    z_va, z_va_y = _ids(root / "ZuCo_SST_data/valid.csv")
    z_te, z_te_y = _ids(root / "ZuCo_SST_data/test.csv")
    s_tr, s_tr_y = _ids(root / "SST_data/train_full_sst.csv")
    s_va, s_va_y = _ids(root / "SST_data/valid_full_sst.csv")
    s_te, s_te_y = _ids(root / "SST_data/test_full_sst.csv")

    z_sets = (set(z_tr), set(z_va), set(z_te))
    s_sets = (set(s_tr), set(s_va), set(s_te))

    print("ZuCo stored split (NOT used by model_ZuCo_SST.py — that script does 5-fold CV):")
    print(
        md_table(
            ["split", "n", "neg", "neu", "pos", "neg share"],
            [
                ["train", len(z_tr), *_mix(z_tr_y), _mix(z_tr_y)[0] / len(z_tr)],
                ["valid", len(z_va), *_mix(z_va_y), _mix(z_va_y)[0] / len(z_va)],
                ["test", len(z_te), *_mix(z_te_y), _mix(z_te_y)[0] / len(z_te)],
                ["all 400", 400, 123, 137, 140, 123 / 400],
            ],
        )
    )
    print(
        f"disjoint: train∩valid={len(z_sets[0] & z_sets[1])} "
        f"train∩test={len(z_sets[0] & z_sets[2])} "
        f"valid∩test={len(z_sets[1] & z_sets[2])} "
        f"union={len(z_sets[0] | z_sets[1] | z_sets[2])}"
    )
    print()
    print("Full SST stored split (used by model_full_SST.py):")
    print(
        md_table(
            ["split", "n", "neg", "neu", "pos"],
            [
                ["train", len(s_tr), *_mix(s_tr_y)],
                ["valid", len(s_va), *_mix(s_va_y)],
                ["test", len(s_te), *_mix(s_te_y)],
            ],
        )
    )
    print(
        f"disjoint: train∩valid={len(s_sets[0] & s_sets[1])} "
        f"train∩test={len(s_sets[0] & s_sets[2])} "
        f"valid∩test={len(s_sets[1] & s_sets[2])} "
        f"union={len(s_sets[0] | s_sets[1] | s_sets[2])}"
    )

    last = FULL_SST_TEST_ROWS % FULL_SST_BATCH
    print()
    print("Trainer errata (documented, not patched):")
    print(f"  test rows {FULL_SST_TEST_ROWS} / batch {FULL_SST_BATCH} → last batch {last}")
    print(
        "  model_full_SST.py assigns `all_preds = preds.cpu().numpy()` each batch, "
        f"so the printed test score is {last}/{FULL_SST_TEST_ROWS} rows "
        f"({last / FULL_SST_TEST_ROWS:.1%}). Validation still extend()s."
    )
    print("  model_ZuCo_SST.py ignores train.csv/valid.csv/test.csv and runs StratifiedKFold.")
    print("  get_matfiles() uses a Windows path \\\\ZuCo_mat_data\\\\.")
    print("  read_ZuCo_mat.py / get_average_sentence_level.py write et_csv_data/, not ZuCo_et_csv_data/.")

    if last != FULL_SST_LAST_BATCH:
        return 1
    if len(z_sets[0] & z_sets[1]) or len(s_sets[0] & s_sets[1]):
        return 1
    if len(z_sets[0] | z_sets[1] | z_sets[2]) != 400:
        return 1
    if len(s_sets[0] | s_sets[1] | s_sets[2]) != 11853:
        return 1
    # The 40-row ZuCo valid split is the unstratified tell: 7 negatives vs ~12 expected.
    if _mix(z_va_y) != (7, 14, 19):
        print("ZuCo valid mix drifted", _mix(z_va_y), file=sys.stderr)
        return 1

    write_text(
        root / "examples/output/07_split_and_errata.md",
        "\n".join(
            [
                "# Splits and trainer errata",
                "",
                f"ZuCo valid mix (neg/neu/pos) = {_mix(z_va_y)} on 40 rows.",
                f"Full SST last batch = {last} of {FULL_SST_TEST_ROWS}.",
                "",
            ]
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
