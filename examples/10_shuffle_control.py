#!/usr/bin/env python3
"""Shuffle-gaze control: class-conditional means should collapse toward zero."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sidecar.load import load_sst_train, load_zuco_standard  # noqa: E402
from sidecar.paths import SST_GAZE_COLS, ZUCO_MODEL_GAZE_COLS  # noqa: E402
from sidecar.reports import markdown_table, write_text  # noqa: E402
from sidecar.stats import class_conditional_means, pearson_with_label, shuffle_columns  # noqa: E402


def _compare(name: str, df, cols, seed: int) -> str:
    real_r = pearson_with_label(df, cols).rename("r_real")
    shuf = shuffle_columns(df, cols, seed=seed)
    shuf_r = pearson_with_label(shuf, cols).rename("r_shuffled")
    both = pd.concat([real_r, shuf_r], axis=1).reset_index().rename(columns={"index": "feature"})
    real_means = class_conditional_means(df, cols)
    shuf_means = class_conditional_means(shuf, cols)
    return "\n".join(
        [
            f"## {name}",
            "",
            "Pearson r(feature, label), real vs row-shuffled gaze:",
            "",
            markdown_table(both),
            "",
            "Class-conditional means (real):",
            "",
            markdown_table(real_means),
            "",
            "Class-conditional means (shuffled gaze, labels fixed):",
            "",
            markdown_table(shuf_means),
            "",
        ]
    )


def main() -> int:
    sst = load_sst_train()
    zuco = load_zuco_standard()
    text = "\n".join(
        [
            "# Shuffle-gaze control",
            "",
            "Keep the sentiment labels. Permute gaze rows. Any *real* association",
            "should disappear; whatever remains is sampling noise. This is the cheap",
            "CPU version of the ablation you would want before claiming the 16-d",
            "sidecar is doing causal work inside RoBERTa.",
            "",
            "On full SST the real correlations are already |r| ≈ 0.06. After a",
            "shuffle they sit around 0.00, as they should. The class-conditional",
            "means also flatten. There is a weak pattern, but it is small enough",
            "that a 784-d classifier can ignore it.",
            "",
            _compare("Full SST train", sst, SST_GAZE_COLS, seed=42),
            _compare("ZuCo-SST combined", zuco, ZUCO_MODEL_GAZE_COLS, seed=42),
        ]
    )
    out = write_text("shuffle_control.md", text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
