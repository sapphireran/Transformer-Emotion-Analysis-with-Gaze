#!/usr/bin/env python3
"""Show that the 5-d full-SST sidecar is almost rank-2 (mostly rank-1)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sidecar.load import load_sst_train, load_zuco_standard  # noqa: E402
from sidecar.paths import SST_GAZE_COLS, ZUCO_MODEL_GAZE_COLS  # noqa: E402
from sidecar.rank import collinearity_matrix, gaze_pca  # noqa: E402
from sidecar.reports import markdown_table, write_text  # noqa: E402


def _pca_block(title: str, df, cols) -> str:
    pca = gaze_pca(df, cols)
    explained = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(len(cols))],
            "explained": pca["explained_variance_ratio"],
            "cumulative": pca["cumulative"],
            "singular": pca["singular_values"],
        }
    )
    loadings = pca["loadings"].reset_index().rename(columns={"index": "component"})
    return "\n".join(
        [
            f"## {title}",
            "",
            f"n = {pca['n']}, features = {list(cols)}",
            "",
            markdown_table(explained),
            "",
            "Loadings (rows are principal axes, columns are original features):",
            "",
            markdown_table(loadings),
            "",
            "Correlation matrix:",
            "",
            markdown_table(collinearity_matrix(df, cols).reset_index().rename(columns={"index": "feature"})),
            "",
        ]
    )


def main() -> int:
    sst = load_sst_train()
    zuco = load_zuco_standard()
    text = "\n".join(
        [
            "# Gaze sidecar rank",
            "",
            "The training models always feed **five** gaze scalars into `nn.Linear(5, 16)`.",
            "On the committed full-SST train table those five columns are almost a single",
            "direction: nFix, TRT, FFD, and GPT correlate at r > 0.98. GD is the only",
            "column with a distinct residual (PC2).",
            "",
            "A 16-d linear sidecar is therefore **overcomplete**. Unless the projection",
            "learns to isolate the GD residual, most of those 16 hidden units are",
            "rotated copies of one reading-time axis.",
            "",
            _pca_block("Full SST train (z-scored sentence gaze)", sst, SST_GAZE_COLS),
            _pca_block(
                "ZuCo-SST combined, the five columns the model actually consumes",
                zuco,
                ZUCO_MODEL_GAZE_COLS,
            ),
            "ZuCo still has TRT ≈ nFixations ≈ GPT, but FFD vs SFD is only weakly",
            "related (SFD is *not* in the five-column model input). Mean pupil size",
            "and omission rate — also unused by `EyeTrackingModel` — are the more",
            "independent physiological channels sitting in the CSV.",
            "",
        ]
    )
    out = write_text("gaze_rank.md", text)
    pca = gaze_pca(sst, SST_GAZE_COLS)
    print(f"wrote {out}")
    print("SST PC1 explained:", float(pca["explained_variance_ratio"][0]))
    print("SST PC2 explained:", float(pca["explained_variance_ratio"][1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
