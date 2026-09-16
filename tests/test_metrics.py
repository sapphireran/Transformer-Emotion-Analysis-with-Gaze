"""Last-batch overwrite vs full concatenation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from sidecar.load import load_sst_test  # noqa: E402
from sidecar.metrics import (  # noqa: E402
    batch_slices,
    last_batch_predictions,
    majority_baseline,
    weighted_scores,
)


class BatchSliceTests(unittest.TestCase):
    def test_1186_by_256(self) -> None:
        slices = batch_slices(1186, 256)
        self.assertEqual(len(slices), 5)
        self.assertEqual(slices[-1], slice(1024, 1186))
        self.assertEqual(slices[-1].stop - slices[-1].start, 162)

    def test_last_batch_helper(self) -> None:
        preds = np.arange(1186)
        last = last_batch_predictions(preds, 256)
        np.testing.assert_array_equal(last, np.arange(1024, 1186))


class MajorityBaselineTests(unittest.TestCase):
    def test_balanced_three_way(self) -> None:
        y = np.array([0, 1, 2, 0, 1, 2])
        maj = majority_baseline(y)
        self.assertAlmostEqual(maj["majority_prior"], 1 / 3)
        self.assertAlmostEqual(maj["accuracy"], 1 / 3)

    def test_weighted_scores_perfect(self) -> None:
        y = np.array([0, 1, 2, 1])
        scores = weighted_scores(y, y)
        self.assertEqual(scores["accuracy"], 1.0)
        self.assertEqual(scores["f1_weighted"], 1.0)


class TestSplitLastBatchPriorTests(unittest.TestCase):
    def test_committed_test_split_last_batch_prior_differs(self) -> None:
        y = load_sst_test()["sentiment_label"].to_numpy()
        self.assertEqual(len(y), 1186)
        last = y[1024:]
        full_pos = float((y == 2).mean())
        last_pos = float((last == 2).mean())
        # Not a flaky equality: we only require the reporting window is different
        # in at least one class count, which is true of this committed shuffle.
        self.assertEqual(len(last), 162)
        self.assertNotEqual(list(np.bincount(y, minlength=3)), list(np.bincount(last, minlength=3)))
        self.assertNotEqual(full_pos, last_pos)


if __name__ == "__main__":
    unittest.main()
