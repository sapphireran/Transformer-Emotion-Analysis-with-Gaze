#!/usr/bin/env python3
"""Split fingerprints: id-disjoint, tiny text leaks, unused ZuCo 80/10/10 files."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sidecar.load import (  # noqa: E402
    load_sst_combined,
    load_sst_test,
    load_sst_train,
    load_sst_valid,
    load_zuco_split,
    load_zuco_standard,
)
from sidecar.reports import markdown_table, write_text  # noqa: E402


def _overlap(a: pd.Series, b: pd.Series) -> int:
    return len(set(a) & set(b))


def main() -> int:
    train, valid, test, comb = load_sst_train(), load_sst_valid(), load_sst_test(), load_sst_combined()
    zuco = load_zuco_standard()
    zt, zv, zte = load_zuco_split("train"), load_zuco_split("valid"), load_zuco_split("test")

    sst_rows = pd.DataFrame(
        [
            {
                "pair": "train ∩ valid",
                "id_overlap": _overlap(train.sentence_id, valid.sentence_id),
                "text_overlap": _overlap(train.sentence, valid.sentence),
            },
            {
                "pair": "train ∩ test",
                "id_overlap": _overlap(train.sentence_id, test.sentence_id),
                "text_overlap": _overlap(train.sentence, test.sentence),
            },
            {
                "pair": "valid ∩ test",
                "id_overlap": _overlap(valid.sentence_id, test.sentence_id),
                "text_overlap": _overlap(valid.sentence, test.sentence),
            },
        ]
    )
    zuco_rows = pd.DataFrame(
        [
            {
                "pair": "train ∩ valid",
                "id_overlap": _overlap(zt.sentence_id, zv.sentence_id),
                "text_overlap": _overlap(zt.sentence, zv.sentence),
            },
            {
                "pair": "train ∩ test",
                "id_overlap": _overlap(zt.sentence_id, zte.sentence_id),
                "text_overlap": _overlap(zt.sentence, zte.sentence),
            },
            {
                "pair": "valid ∩ test",
                "id_overlap": _overlap(zv.sentence_id, zte.sentence_id),
                "text_overlap": _overlap(zv.sentence, zte.sentence),
            },
        ]
    )

    tv = sorted(set(train.sentence) & set(valid.sentence))
    tt = sorted(set(train.sentence) & set(test.sentence))
    corpus_overlap = sorted(set(zuco.sentence) & set(comb.sentence))

    train_dups = int(train.sentence.duplicated().sum())
    comb_dups = int(comb.sentence.duplicated().sum())

    text = "\n".join(
        [
            "# Split fingerprint",
            "",
            "## Full SST (`SST_data/spilt.py`, seed 42, 80/10/10)",
            "",
            f"train {len(train)} + valid {len(valid)} + test {len(test)} = "
            f"{len(train)+len(valid)+len(test)} (combined {len(comb)}).",
            "Ids partition the combined table. Duplicate *texts* still leak across splits",
            "because `train_test_split` was keyed on rows, not unique strings.",
            "",
            markdown_table(sst_rows, floatfmt="{:.0f}"),
            "",
            f"Duplicate sentence strings inside train: {train_dups} extra rows "
            f"({train.sentence.nunique()} unique / {len(train)} rows).",
            f"Combined table has {comb_dups} extra duplicate-text rows "
            f"({comb.sentence.nunique()} unique / {len(comb)}).",
            "",
            "Train/valid shared text:",
            "",
            *([f"- `{s}`" for s in tv] or ["- (none)"]),
            "",
            "Train/test shared text:",
            "",
            *([f"- `{s}`" for s in tt] or ["- (none)"]),
            "",
            "## ZuCo-SST 80/10/10 CSVs vs what the trainer actually does",
            "",
            f"`ZuCo_SST_data/{{train,valid,test}}.csv` is {len(zt)}/{len(zv)}/{len(zte)}.",
            "`model_ZuCo_SST.py` **ignores those files** and runs `StratifiedKFold(5)` on",
            "`combined_sst_et_standard.csv` instead. The 40-row valid split is also",
            "badly imbalanced (only 7 negatives), which is one reason a stratified",
            "k-fold on all 400 rows is the more defensible protocol — if you remember",
            "that those CSVs are leftover from an earlier split script (`spilt.py`).",
            "",
            markdown_table(zuco_rows, floatfmt="{:.0f}"),
            "",
            "## Two almost-disjoint corpora",
            "",
            f"Only **{len(corpus_overlap)}** review strings appear in both the 400-row ZuCo",
            "table and the 11.8k full-SST table. Measured gaze and predicted gaze are",
            "not two views of the same items.",
            "",
            *[f"- `{s}`" for s in corpus_overlap],
            "",
        ]
    )
    out = write_text("split_fingerprint.md", text)
    print(f"wrote {out}")
    print("sst text leaks", len(tv), len(tt))
    print("corpus overlap", len(corpus_overlap))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
