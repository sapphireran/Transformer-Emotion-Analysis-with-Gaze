#!/usr/bin/env python3
"""Dummy late-fusion forward pass — locks the 5→16→784→3 shapes from the docs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))

from gazekit.fusion import (
    DROPOUT_P,
    ENCODER_HIDDEN,
    FUSED_WIDTH,
    GAZE_HIDDEN,
    GAZE_IN,
    NUM_LABELS,
    GazeFusionForward,
    assert_architecture_constants,
    shapes_report,
)
from gazekit.io import gaze_matrix, load_sentence_table
from gazekit.paths import default_paths
from gazekit.report import section


def build_report(root: Path | None = None, batch_size: int = 8) -> dict:
    assert_architecture_constants()
    shapes = shapes_report(batch_size=batch_size)
    paths = default_paths(root)
    df = load_sentence_table(paths.zuco_combined_standard)
    et = gaze_matrix(df)[:batch_size]
    import numpy as np

    rng = np.random.default_rng(0)
    pooled = rng.normal(0, 1, (et.shape[0], ENCODER_HIDDEN))
    logits = GazeFusionForward(seed=0)(pooled, et)
    return {
        "constants": {
            "encoder_hidden": ENCODER_HIDDEN,
            "gaze_in": GAZE_IN,
            "gaze_hidden": GAZE_HIDDEN,
            "fused_width": FUSED_WIDTH,
            "num_labels": NUM_LABELS,
            "dropout_p": DROPOUT_P,
        },
        "shapes": {k: tuple(v) for k, v in shapes.items()},
        "real_et_batch": et.shape,
        "logits_from_real_et": logits.shape,
        "logits_finite": bool(np.isfinite(logits).all()),
    }


def render(report: dict) -> str:
    const_lines = "\n".join(f"  {k}={v}" for k, v in report["constants"].items())
    shape_lines = "\n".join(f"  {k}: {v}" for k, v in report["shapes"].items())
    return "\n".join(
        [
            section("Architecture constants (EyeTrackingModel)", const_lines),
            section("Dummy batch shapes", shape_lines),
            section(
                "Forward on real ZuCo ET rows + random pooler states",
                f"et batch {report['real_et_batch']}\n"
                f"logits {report['logits_from_real_et']}\n"
                f"finite={report['logits_finite']}",
            ),
        ]
    )


def main() -> None:
    print(render(build_report()))


if __name__ == "__main__":
    main()
