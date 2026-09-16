"""Replay EyeTrackingModel concat fusion with NumPy (no BERT download)."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import sys

import numpy as np

from teag_examples.fusion import FusionConfig, NumpyEyeTrackingFusion, cross_entropy, softmax
from teag_examples.io import load_full_sst_combined, load_zuco_combined
from teag_examples.paths import docs_assets, examples_output
from teag_examples.reports import write_json, write_markdown
from teag_examples.schema import FULL_SST_GAZE_COLS, LABEL_NAMES, ZUCO_MODEL_GAZE_COLS
from teag_examples.viz import save_fusion_shapes


def _batch_from_frame(df, columns, n: int, rng: np.random.Generator, hidden: int):
    gaze = df.loc[: n - 1, list(columns)].to_numpy(dtype=np.float64)
    pooled = rng.normal(0.0, 0.02, size=(n, hidden))
    labels = df.loc[: n - 1, "sentiment_label"].to_numpy(dtype=int)
    return pooled, gaze, labels


def main(argv: list[str] | None = None) -> int:
    del argv
    cfg = FusionConfig()
    rng = np.random.default_rng(7)
    model = NumpyEyeTrackingFusion(cfg, rng=rng)

    zuco = load_zuco_combined("standard")
    sst = load_full_sst_combined()
    n = 8
    pooled_z, gaze_z, y_z = _batch_from_frame(zuco, ZUCO_MODEL_GAZE_COLS, n, rng, cfg.hidden_size)
    pooled_s, gaze_s, y_s = _batch_from_frame(sst, FULL_SST_GAZE_COLS, n, rng, cfg.hidden_size)

    logits_z = model.forward(pooled_z, gaze_z, train=False)
    logits_zero = model.forward(pooled_z, np.zeros_like(gaze_z), train=False)
    delta = np.abs(logits_z - logits_zero).mean(axis=0)

    # Same pooled vector, two real gaze rows — does the gaze branch flip the argmax?
    flips = []
    for i in range(n):
        for j in range(i + 1, n):
            a = model.predict(pooled_z[i], gaze_z[i])[0]
            b = model.predict(pooled_z[i], gaze_z[j])[0]
            if a != b:
                flips.append((i, j, int(a), int(b)))

    train_a = model.forward(pooled_z, gaze_z, train=True, dropout_rng=np.random.default_rng(1))
    train_b = model.forward(pooled_z, gaze_z, train=True, dropout_rng=np.random.default_rng(2))
    eval_a = model.forward(pooled_z, gaze_z, train=False)
    eval_b = model.forward(pooled_z, gaze_z, train=False)

    ce = cross_entropy(logits_z, y_z)
    probs = softmax(logits_z)
    save_fusion_shapes(docs_assets() / "fusion_shapes.png")

    payload = {
        "config": cfg.__dict__,
        "concat_width": model.concat_width,
        "zuco_logits_row0": logits_z[0].tolist(),
        "zuco_probs_row0": probs[0].tolist(),
        "mean_abs_logit_shift_vs_zero_gaze": delta.tolist(),
        "gaze_pair_argmax_flips": flips[:12],
        "n_gaze_pair_flips": len(flips),
        "dropout_train_diverges": bool(np.max(np.abs(train_a - train_b)) > 1e-12),
        "eval_is_deterministic": bool(np.allclose(eval_a, eval_b)),
        "cross_entropy_random_pooled_zuco": ce,
        "label_names": dict(LABEL_NAMES),
        "sst_gaze_row0": gaze_s[0].tolist(),
        "true_labels_zuco_batch": y_z.tolist(),
        "true_labels_sst_batch": y_s.tolist(),
    }

    md = f"""# Fusion forward-pass demo

NumPy clone of `EyeTrackingModel` (random pooled vectors, **real** gaze rows).

- concat width = {model.concat_width} (768 + 16)
- mean |Δ logit| vs zero gaze = {np.round(delta, 4).tolist()}
- pairs of ZuCo gaze vectors that flip argmax on a fixed pooled vector: {len(flips)}
- train dropout diverges across RNG: {payload['dropout_train_diverges']}
- eval forward is deterministic: {payload['eval_is_deterministic']}
- cross-entropy on this random-pooled batch (not a trained score): {ce:.4f}

Pooled text is Gaussian noise so class probabilities are not meaningful as
sentiment accuracy. The demo only shows that the gaze branch is wired the same
way as the PyTorch module: linear 5→16, concat, dropout, linear 784→3.
"""
    write_markdown(md, examples_output() / "fusion_demo.md")
    write_markdown(md, docs_assets() / "fusion_demo.md")
    write_json(payload, examples_output() / "fusion_demo.json")
    sys.stdout.write(md + "\n")
    sys.stdout.write(json_preview(payload) + "\n")
    return 0


def json_preview(payload: dict) -> str:
    interesting = {
        k: payload[k]
        for k in (
            "concat_width",
            "mean_abs_logit_shift_vs_zero_gaze",
            "n_gaze_pair_flips",
            "dropout_train_diverges",
            "eval_is_deterministic",
        )
    }
    return str(interesting)


if __name__ == "__main__":
    raise SystemExit(main())
