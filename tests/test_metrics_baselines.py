"""Metrics and a separable ridge-fusion smoke test."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gazebook.baselines import (
    hash_bow,
    majority_predict,
    ridge_ovr_fit,
    ridge_ovr_predict,
    stratified_kfold,
    tokenize,
)
from gazebook.metrics import accuracy, confusion, weighted_scores
from gazebook.stats import condition_number_corr, minmax, pearson, zscore


class MetricTests(unittest.TestCase):
    def test_perfect(self):
        y = np.array([0, 1, 2, 2])
        s = weighted_scores(y, y)
        self.assertEqual(s.accuracy, 1.0)
        self.assertEqual(s.f1, 1.0)
        self.assertEqual(s.precision, 1.0)
        self.assertEqual(s.recall, 1.0)

    def test_majority_all_one_class(self):
        y = np.array([1, 1, 1, 1])
        p = np.array([1, 1, 1, 1])
        self.assertEqual(accuracy(y, p), 1.0)

    def test_confusion_shape(self):
        y = np.array([0, 1, 2])
        p = np.array([0, 2, 2])
        mat = confusion(y, p)
        self.assertEqual(mat.shape, (3, 3))
        self.assertEqual(int(mat[1, 2]), 1)


class BaselineTests(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("Slow, silly!"), ["slow", "silly"])

    def test_hash_stable(self):
        a = hash_bow(["hello world", "hello"], dim=32)
        b = hash_bow(["hello world", "hello"], dim=32)
        self.assertTrue(np.allclose(a, b))
        self.assertEqual(a.shape, (2, 32))

    def test_majority(self):
        y = np.array([0, 2, 2, 2, 1])
        pred = majority_predict(y, 4)
        self.assertTrue(np.all(pred == 2))

    def test_stratified_coverage(self):
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)
        seen = []
        for train, test in stratified_kfold(y, n_splits=5, seed=0):
            self.assertEqual(len(train) + len(test), 30)
            seen.extend(test.tolist())
            # each fold should see every class
            self.assertEqual(set(y[test]), {0, 1, 2})
        self.assertEqual(sorted(seen), list(range(30)))

    def test_separable_fusion(self):
        rng = np.random.default_rng(0)
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])
        text = rng.normal(size=(len(y), 6))
        gaze = rng.normal(size=(len(y), 5))
        # Plant a linear signal in the concat space.
        for i, label in enumerate(y):
            text[i, label] += 4.0
            gaze[i, 0] += label
        X = np.concatenate([text, gaze], axis=1)
        W = ridge_ovr_fit(X, y, l2=0.1)
        pred = ridge_ovr_predict(W, X)
        self.assertGreaterEqual(accuracy(y, pred), 0.9)


class StatsTests(unittest.TestCase):
    def test_pearson_self(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(pearson(x, x), 1.0)

    def test_minmax_bounds(self):
        X = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
        M = minmax(X)
        self.assertAlmostEqual(M[:, 0].min(), 0.0)
        self.assertAlmostEqual(M[:, 0].max(), 1.0)

    def test_zscore_mean(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Z = zscore(X, ddof=0)
        self.assertTrue(np.allclose(Z.mean(axis=0), 0.0))

    def test_condition_identity(self):
        X = np.eye(5)
        # 5 orthogonal columns → corr is identity → cond = 1
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 3))
        # make columns orthogonal-ish via QR
        Q, _ = np.linalg.qr(X)
        cond = condition_number_corr(Q)
        self.assertLess(cond, 1.1)


if __name__ == "__main__":
    unittest.main()
