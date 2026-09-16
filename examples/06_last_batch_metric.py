#!/usr/bin/env python3
"""Reproduce the model_full_SST.py test-loop overwrite (last batch only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sidecar.load import load_sst_test  # noqa: E402
from sidecar.metrics import (  # noqa: E402
    batch_slices,
    full_predictions,
    last_batch_predictions,
    majority_baseline,
    weighted_scores,
)
from sidecar.reports import markdown_table, write_text  # noqa: E402

BATCH_SIZE = 256  # as in model_full_SST.py


def main() -> int:
    test = load_sst_test()
    y = test["sentiment_label"].to_numpy()
    n = len(y)
    slices = batch_slices(n, BATCH_SIZE)
    last = slices[-1]

    # Constant-majority predictor: the last batch prior differs from the full set,
    # so last-batch accuracy is already the wrong number even before a real model.
    maj = majority_baseline(y)
    pred_maj = np.full(n, maj["majority_label"])

    rng = np.random.default_rng(0)
    # A second toy predictor: 70% correct, shuffled — last batch should jitter.
    pred_noisy = y.copy()
    flip = rng.random(n) > 0.70
    pred_noisy[flip] = rng.integers(0, 3, size=int(flip.sum()))

    rows = []
    for name, pred in [("majority", pred_maj), ("70pct_correct_toy", pred_noisy)]:
        full = weighted_scores(y, full_predictions(pred))
        last_scores = weighted_scores(y[last], last_batch_predictions(pred, BATCH_SIZE))
        rows.append(
            {
                "predictor": name,
                "eval": "full_test",
                **{k: v for k, v in full.items() if k != "n"},
                "n": full["n"],
            }
        )
        rows.append(
            {
                "predictor": name,
                "eval": "last_batch_only",
                **{k: v for k, v in last_scores.items() if k != "n"},
                "n": last_scores["n"],
            }
        )

    last_labels = pd.Series(y[last]).value_counts().sort_index()
    full_labels = pd.Series(y).value_counts().sort_index()
    last_n = last.stop - last.start
    prior = pd.DataFrame(
        {
            "label": [0, 1, 2],
            "full_n": [int(full_labels.get(i, 0)) for i in range(3)],
            "full_share": [float(full_labels.get(i, 0)) / n for i in range(3)],
            "last_n": [int(last_labels.get(i, 0)) for i in range(3)],
            "last_share": [float(last_labels.get(i, 0)) / last_n for i in range(3)],
        }
    )

    text = "\n".join(
        [
            "# Last-batch test metric (as committed)",
            "",
            "In `model_full_SST.py` the test loop does:",
            "",
            "```python",
            "preds = torch.argmax(logits, dim=1)",
            "all_preds = preds.cpu().numpy()   # assignment, not extend",
            "all_labels = labels.cpu().numpy()",
            "```",
            "",
            "The validation loop correctly `extend`s. The test loop therefore reports",
            f"metrics on the **last DataLoader batch only**. Committed test size {n},",
            f"`batch_size = {BATCH_SIZE}`, so that is rows `{last.start}:{last.stop}`",
            f"({last_n} examples, {last_n/n:.1%} of the test set). The printed",
            "`Test Acc` is not a test-set score.",
            "",
            "The comment above the checkpoint also says 'best F1' while the `if`",
            "condition is `val_acc > best_val_acc`. Two naming bugs in the same file.",
            "",
            f"Number of batches: {len(slices)} (the first {len(slices)-1} are size {BATCH_SIZE}).",
            "",
            "## Label prior: full test vs last batch",
            "",
            markdown_table(prior),
            "",
            "## Toy predictors: full vs last-batch scores",
            "",
            "These are not model results. They show that the *reporting procedure*",
            "moves the number even when the predictor is trivial.",
            "",
            markdown_table(pd.DataFrame(rows)),
            "",
        ]
    )
    out = write_text("last_batch_metric.md", text)
    print(f"wrote {out}")
    print("n", n, "last", last, "n_batches", len(slices))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
