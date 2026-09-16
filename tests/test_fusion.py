"""Unit tests for the gaze-sidecar fusion geometry (no torch, no CSVs)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from sidecar.fusion import FUSION_CONCAT_DIM, SidecarFusion, softmax  # noqa: E402


class FusionGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SidecarFusion.from_seed(seed=0)
        rng = np.random.default_rng(1)
        self.pooler = rng.normal(size=(8, 768))
        self.gaze = rng.normal(size=(8, 5))

    def test_concat_is_784(self) -> None:
        h = self.model.concat(self.pooler, self.gaze)
        self.assertEqual(h.shape, (8, FUSION_CONCAT_DIM))
        self.assertEqual(self.model.concat_dim, 784)

    def test_logits_are_three_way(self) -> None:
        logits = self.model.logits(self.pooler, self.gaze)
        self.assertEqual(logits.shape, (8, 3))
        preds = self.model.predict(self.pooler, self.gaze)
        self.assertEqual(preds.shape, (8,))
        self.assertTrue(set(preds.tolist()) <= {0, 1, 2})

    def test_eval_dropout_is_identity(self) -> None:
        a = self.model.logits(self.pooler, self.gaze, training=False)
        b = self.model.logits(self.pooler, self.gaze, training=False)
        np.testing.assert_allclose(a, b)

    def test_train_dropout_changes_logits(self) -> None:
        off = self.model.logits(self.pooler, self.gaze, training=False)
        on = self.model.logits(self.pooler, self.gaze, training=True)
        self.assertGreater(np.abs(on - off).mean(), 0.0)

    def test_gaze_projection_is_linear(self) -> None:
        eye = self.model.project_gaze(self.gaze)
        self.assertEqual(eye.shape, (8, 16))
        # Doubling gaze doubles the projection (zero bias).
        eye2 = self.model.project_gaze(2 * self.gaze)
        np.testing.assert_allclose(eye2, 2 * eye)

    def test_softmax_rows_sum_to_one(self) -> None:
        probs = softmax(self.model.logits(self.pooler, self.gaze))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(8), atol=1e-12)

    def test_wrong_gaze_dim_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.model.project_gaze(np.zeros((8, 4)))

    def test_single_row_accepted(self) -> None:
        h = self.model.concat(self.pooler[0], self.gaze[0])
        self.assertEqual(h.shape, (1, 784))


if __name__ == "__main__":
    unittest.main()
