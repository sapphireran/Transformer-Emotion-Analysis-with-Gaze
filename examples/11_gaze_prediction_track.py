#!/usr/bin/env python3
"""Gaze-prediction track: placeholder zeros, Provo, and predicted slices."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sidecar.paths import (  # noqa: E402
    PREDICTION_TEST,
    PREDICTION_TEST_V2,
    PROVO,
    ROOT,
    SST_ET_PLACEHOLDER,
)
from sidecar.reports import markdown_table, write_text  # noqa: E402


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> int:
    pred = pd.read_csv(PREDICTION_TEST)
    placeholder = pd.read_csv(SST_ET_PLACEHOLDER)
    provo = pd.read_csv(PROVO)
    pred_sents = sorted(pred.sentence_id.unique())

    text = "\n".join(
        [
            "# Gaze-prediction track",
            "",
            "The `result/*.png` scatter/histogram grids are in the **raw millisecond /**",
            "**count** space used by `gaze_prediction/data/*.csv` and Provo, *not* the",
            "z-scored `SST_data/*_full_sst.csv` tables (those have mean ≈ 0, std ≈ 1).",
            "Do not paste a z-score next to those plots and call it the same feature.",
            "",
            f"## `{_rel(SST_ET_PLACEHOLDER)}`",
            "",
            "Word-tokenized SST with gaze columns filled by zeros. This is the blank",
            "form a predictor is supposed to fill. Committed file:",
            f"{len(placeholder)} word rows, {placeholder.sentence_id.nunique()} sentences,",
            f"nFix all zero? {(placeholder['nFix'] == 0).all()}.",
            "",
            f"## `{_rel(PREDICTION_TEST)}`",
            "",
            f"{len(pred)} word rows covering **{pred.sentence_id.nunique()} sentences**",
            f"with ids {int(pred.sentence_id.min())}…{int(pred.sentence_id.max())}.",
            "That is the last 100 ZuCo-SST reviews, not the 11.8k full SST test split.",
            "nFix here lives on a 0–50 scale (mean "
            f"{pred.nFix.mean():.2f}), matching the train/test PNG histograms.",
            "",
            f"## `{_rel(PREDICTION_TEST_V2)}`",
            "",
            "Large sister table (same schema as the zero placeholder). Used as the",
            "word-level dump of predicted gaze over the whole SST token stream.",
            "",
            f"## `{_rel(PROVO)}`",
            "",
            f"{len(provo)} word rows, {provo.sentence_id.nunique()} sentences, extra",
            "`fixProp` column that the ZuCo-style five-tuple does not have.",
            "",
            "### prediction_test feature means",
            "",
            markdown_table(
                pd.DataFrame(
                    {
                        "feature": ["nFix", "FFD", "GPT", "TRT", "GD"],
                        "mean": [pred[c].mean() for c in ["nFix", "FFD", "GPT", "TRT", "GD"]],
                        "std": [pred[c].std() for c in ["nFix", "FFD", "GPT", "TRT", "GD"]],
                    }
                )
            ),
            "",
            "### provo feature means",
            "",
            markdown_table(
                pd.DataFrame(
                    {
                        "feature": ["nFix", "FFD", "GPT", "TRT", "fixProp"],
                        "mean": [provo[c].mean() for c in ["nFix", "FFD", "GPT", "TRT", "fixProp"]],
                        "std": [provo[c].std() for c in ["nFix", "FFD", "GPT", "TRT", "fixProp"]],
                    }
                )
            ),
            "",
            f"prediction_test sentence ids (n={len(pred_sents)}): {pred_sents[0]}…{pred_sents[-1]}",
            "",
        ]
    )
    out = write_text("gaze_prediction_track.md", text)
    print(f"wrote {out}")
    print("placeholder sents", placeholder.sentence_id.nunique(), "pred sents", pred.sentence_id.nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
