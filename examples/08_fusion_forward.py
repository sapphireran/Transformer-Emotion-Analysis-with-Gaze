#!/usr/bin/env python3
"""NumPy forward pass for the 768+16 concat used by EyeTrackingModel."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sidecar.fusion import FUSION_CONCAT_DIM, SidecarFusion, softmax  # noqa: E402
from sidecar.load import load_sst_train  # noqa: E402
from sidecar.paths import SST_GAZE_COLS  # noqa: E402
from sidecar.reports import markdown_table, write_text  # noqa: E402


def main() -> int:
    model = SidecarFusion.from_seed(seed=7)
    train = load_sst_train()
    gaze = train.loc[:4, list(SST_GAZE_COLS)].to_numpy(dtype=float)
    rng = np.random.default_rng(7)
    pooler = rng.normal(0.0, 0.02, size=(len(gaze), 768))  # fake [CLS]/<s> pooler
    h = model.concat(pooler, gaze)
    logits = model.logits(pooler, gaze, training=False)
    probs = softmax(logits)
    preds = model.predict(pooler, gaze)
    eye = model.project_gaze(gaze)

    preview = pd.DataFrame(
        {
            "row": np.arange(len(gaze)),
            "nFix": gaze[:, 0],
            "GD": gaze[:, 1],
            "eye_l2": np.linalg.norm(eye, axis=1),
            "concat_dim": h.shape[1],
            "p_neg": probs[:, 0],
            "p_neu": probs[:, 1],
            "p_pos": probs[:, 2],
            "pred": preds,
            "true_label": train.loc[:4, "sentiment_label"].to_numpy(),
        }
    )

    # Show dropout changes logits in train mode but not the concat geometry.
    logits_train_mode = model.logits(pooler, gaze, training=True)
    delta = np.abs(logits_train_mode - logits).mean()

    text = "\n".join(
        [
            "# Sidecar fusion walkthrough",
            "",
            "This is the geometry of `EyeTrackingModel` without downloading RoBERTa.",
            "Pooler vectors below are random N(0, 0.02) stand-ins so the example stays",
            "CPU-only and offline. Gaze rows are the first five *real* full-SST train",
            "examples.",
            "",
            "```",
            "gaze  (B, 5)  --Linear 5→16, no activation-->  eye (B, 16)",
            "pooler(B, 768) ----------------------------\\",
            "                                           concat (B, 784)",
            "                                           Dropout p=0.1",
            "                                           Linear 784→3 logits",
            "```",
            "",
            f"- concat dim: **{model.concat_dim}** (expected {FUSION_CONCAT_DIM})",
            f"- W_eye shape: {tuple(model.W_eye.shape)}",
            f"- W_cls shape: {tuple(model.W_cls.shape)}",
            f"- mean |Δlogit| when dropout is on vs off (same seed): {delta:.6f}",
            "",
            "The training scripts never apply ReLU/GELU on the gaze branch. The sidecar",
            "is a linear basis expansion of five already-collinear scalars.",
            "",
            markdown_table(preview),
            "",
            "Random poolers plus untrained sidecar weights are **not** a sentiment",
            "model. The table exists to pin shapes and to show how little of the 784-d",
            "vector is gaze (16 / 784 ≈ 2.0%). Any claim that 'the model uses eye",
            "tracking' has to survive an ablation that zeros or shuffles those 16",
            "coordinates.",
            "",
        ]
    )
    out = write_text("fusion_forward.md", text)
    print(f"wrote {out}")
    print("concat", h.shape, "logits", logits.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
