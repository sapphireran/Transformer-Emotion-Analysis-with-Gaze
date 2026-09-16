#!/usr/bin/env python3
"""Show the concat-fusion geometry without downloading BERT or RoBERTa.

The real EyeTrackingModel does:

    pooler_output (768)  ||  Linear(5, 16)  ->  Linear(784, 3)

This stand-in uses a hashed bag-of-words of size 48 so the example can fit
a logistic head on the ZuCo 400 in a second and print a number. That number
is a wiring check, not a paper result.
"""

from __future__ import annotations

import numpy as np

from examples._common import banner
from tea_gaze.io import load_zuco_combined
from tea_gaze.metrics import classification_metrics, majority_baseline
from tea_gaze.models import ToyEyeTrackingFusion, ToyFusionGeometry, hashed_bow_vector
from tea_gaze.schema import (
    HIDDEN_LAYER_SIZE,
    NUM_EYE_TRACKING_FEATURES,
    TRANSFORMER_HIDDEN_SIZE,
    ZUCO_MODEL_FEATURES,
)
from tea_gaze.reports import markdown_table
import pandas as pd


def main() -> None:
    geo_real = ToyFusionGeometry(
        text_dim=TRANSFORMER_HIDDEN_SIZE,
        et_in=NUM_EYE_TRACKING_FEATURES,
        et_hidden=HIDDEN_LAYER_SIZE,
    )
    geo_toy = ToyFusionGeometry()

    banner("Geometry")
    print("Historical EyeTrackingModel:", geo_real.describe())
    print("This example stand-in:     ", geo_toy.describe())

    demo_text = "Slow, silly and unintentionally hilarious."
    demo_et = np.array([[4.13, 2.32, 3.50, 4.04, 1.89]])
    fusion = ToyEyeTrackingFusion(seed=7)
    hidden = fusion.project_eye_tracking(demo_et)
    encoded = fusion.encode([demo_text], demo_et)
    bow = hashed_bow_vector(demo_text)

    banner("One forward (unfitted) on a ZuCo sentence")
    print(f"text: {demo_text!r}")
    print(f"hashed BoW L2={np.linalg.norm(bow):.3f}  nonzero={int(np.count_nonzero(bow))}/{bow.size}")
    print(f"ET hidden (tanh(W x + b)) shape={hidden.shape}  sample={np.round(hidden[0, :4], 3)}")
    print(f"concat vector shape={encoded.shape}  (text {geo_toy.text_dim} + ET {geo_toy.et_hidden})")

    bundle = load_zuco_combined(scaling="standard")
    frame = bundle.frame
    texts = frame["sentence"].tolist()
    et = frame.loc[:, list(ZUCO_MODEL_FEATURES)].to_numpy()
    labels = frame["sentiment_label"].to_numpy()

    # 320/80 split that mirrors ZuCo_SST_data/spilt.py proportions, different use
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(frame))
    train, test = perm[:320], perm[320:]

    fusion.fit(np.take(texts, train), et[train], labels[train])
    preds = fusion.predict(np.take(texts, test), et[test])
    toy = classification_metrics(preds, labels[test])
    _, majority = majority_baseline(labels[test])

    text_only = ToyEyeTrackingFusion(seed=7)
    zeros = np.zeros_like(et)
    text_only.fit(np.take(texts, train), zeros[train], labels[train])
    text_preds = text_only.predict(np.take(texts, test), zeros[test])
    text_metrics = classification_metrics(text_preds, labels[test])

    banner("Holdout (80 rows) on ZuCo standard — toy model only")
    table = pd.DataFrame(
        [
            {"setup": "majority on holdout", **majority.as_dict()},
            {"setup": "hashed BoW + zero ET", **text_metrics.as_dict()},
            {"setup": "hashed BoW + projected ET", **toy.as_dict()},
        ]
    )
    print(markdown_table(table))
    print()
    print("If concat is wired, the last row can move (up or down) vs text-only.")
    print("A large jump here would be suspicious: gaze–label r is ~0.05.")


if __name__ == "__main__":
    main()
