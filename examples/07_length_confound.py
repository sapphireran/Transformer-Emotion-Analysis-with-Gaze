#!/usr/bin/env python3
"""Length confound: predicted gaze on full SST is anti-correlated with token count."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sidecar.load import load_sst_train, load_zuco_standard  # noqa: E402
from sidecar.paths import SST_GAZE_COLS, ZUCO_MODEL_GAZE_COLS  # noqa: E402
from sidecar.reports import markdown_table, write_text  # noqa: E402
from sidecar.stats import residualize, token_count  # noqa: E402


def _block(name: str, df, gaze_cols, text_col="sentence") -> str:
    ntok = token_count(df[text_col]).astype(float)
    rows = []
    for col in gaze_cols:
        y = df[col].to_numpy(dtype=float)
        resid = residualize(y, ntok.to_numpy())
        rows.append(
            {
                "feature": col,
                "r_with_ntokens": float(pd.Series(y).corr(ntok)),
                "r_with_label": float(pd.Series(y).corr(df["sentiment_label"])),
                "r_residual_with_label": float(pd.Series(resid).corr(df["sentiment_label"])),
            }
        )
    stats = pd.DataFrame(rows)
    length_vs_label = float(ntok.corr(df["sentiment_label"]))
    return "\n".join(
        [
            f"## {name}",
            "",
            f"n = {len(df)}, mean tokens = {ntok.mean():.2f}, "
            f"r(ntokens, label) = {length_vs_label:.4f}",
            "",
            markdown_table(stats),
            "",
        ]
    )


def main() -> int:
    sst = load_sst_train()
    zuco = load_zuco_standard()
    text = "\n".join(
        [
            "# Length confound",
            "",
            "On the **predicted / z-scored** full-SST sidecar, nFix, TRT, FFD, and GPT",
            "all correlate around **r ≈ −0.30 to −0.37** with whitespace token count.",
            "GD does not (near zero). Residualizing each feature against token count",
            "barely changes r(feature, label), so the weak sentiment association is",
            "not just 'longer reviews have a polarity'.",
            "",
            "The negative length–gaze correlation is itself a property of the",
            "*imputed* table: measured ZuCo gaze (per-word means aggregated to the",
            "sentence) does not show the same strong anti-correlation. Treat the",
            "full-SST sidecar as a model output, not as a millisecond stopwatch.",
            "",
            _block("Full SST train", sst, SST_GAZE_COLS),
            _block("ZuCo-SST combined (standard scaled)", zuco, ZUCO_MODEL_GAZE_COLS),
        ]
    )
    out = write_text("length_confound.md", text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
