"""Numpy fusion walkthrough stays aligned with EyeTrackingModel shapes."""

from __future__ import annotations

import unittest

import numpy as np

from examples.lib.fusion import FusionWalkthrough, demo_batch, softmax


class FusionWalkthroughTest(unittest.TestCase):
    def test_default_shapes(self) -> None:
        model = FusionWalkthrough(seed=3)
        pooled, gaze = demo_batch(batch_size=8, seed=4)
        out = model.forward(pooled, gaze)
        self.assertEqual(out["pooled"].shape, (8, 768))
        self.assertEqual(out["gaze_hidden"].shape, (8, 16))
        self.assertEqual(out["concat"].shape, (8, 784))
        self.assertEqual(out["logits"].shape, (8, 3))
        self.assertEqual(out["probs"].shape, (8, 3))
        np.testing.assert_allclose(out["probs"].sum(axis=1), np.ones(8), atol=1e-6)

    def test_extra_gaze_channel_widens_concat(self) -> None:
        model = FusionWalkthrough(num_gaze_features=6, gaze_hidden=16, seed=1)
        pooled, gaze = demo_batch(batch_size=2, num_gaze_features=6, seed=2)
        out = model.forward(pooled, gaze)
        self.assertEqual(model.concat_width, 784)
        self.assertEqual(out["gaze_hidden"].shape, (2, 16))

    def test_rejects_width_mismatch(self) -> None:
        model = FusionWalkthrough()
        pooled, _ = demo_batch(batch_size=2)
        with self.assertRaises(ValueError):
            model.forward(pooled, np.zeros((2, 4)))

    def test_softmax_rows_sum_to_one(self) -> None:
        logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        probs = softmax(logits, axis=1)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(2))
        self.assertGreater(probs[0, 2], probs[0, 0])


if __name__ == "__main__":
    unittest.main()
