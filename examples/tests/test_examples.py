"""Unit tests for the docs examples.

These re-run the same checks as the CLI scripts so a single
`python3 -m unittest discover -s examples/tests -v` is enough.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1]
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import fusion_forward  # noqa: E402
import gaze_by_sentiment  # noqa: E402
import inspect_datasets  # noqa: E402
import scaling_check  # noqa: E402
import split_integrity  # noqa: E402
import word_to_sentence  # noqa: E402
from csvutil import read_dicts  # noqa: E402
from paths import (  # noqa: E402
    PRED_V2,
    SST_WORD_ZEROS,
    ZUCO_COMBINED_STD,
    ZUCO_LABELS,
)


class InspectDatasetsTests(unittest.TestCase):
    def test_inventory_matches_docs(self) -> None:
        self.assertEqual(inspect_datasets.main(["--quiet"]), 0)

    def test_zuco_labels_join_to_combined(self) -> None:
        _, labels = read_dicts(ZUCO_LABELS)
        _, combined = read_dicts(ZUCO_COMBINED_STD)
        self.assertEqual(
            [row["sentence_id"] for row in labels],
            [row["sentence_id"] for row in combined],
        )
        self.assertEqual(
            [row["sentiment_label"] for row in labels],
            [row["sentiment_label"] for row in combined],
        )


class SplitIntegrityTests(unittest.TestCase):
    def test_partitions(self) -> None:
        self.assertEqual(split_integrity.main(["--quiet"]), 0)


class ScalingCheckTests(unittest.TestCase):
    def test_minmax_and_zscore(self) -> None:
        self.assertEqual(scaling_check.main(["--quiet"]), 0)


class FusionForwardTests(unittest.TestCase):
    def test_demo_shapes_and_gaze_moves_logits(self) -> None:
        self.assertEqual(fusion_forward.main(["--quiet", "--batch-size", "4"]), 0)

    def test_rejects_bad_gaze_width(self) -> None:
        import numpy as np

        rng = np.random.default_rng(1)
        model = fusion_forward.EyeTrackingFusion(rng)
        pooled = rng.normal(size=(2, fusion_forward.ENCODER_HIDDEN))
        bad_gaze = rng.normal(size=(2, 4))
        with self.assertRaises(ValueError):
            model.forward(pooled, bad_gaze)


class GazeBySentimentTests(unittest.TestCase):
    def test_class_means(self) -> None:
        self.assertEqual(gaze_by_sentiment.main(["--quiet"]), 0)


class WordToSentenceTests(unittest.TestCase):
    def test_subject1_reconstruction(self) -> None:
        self.assertEqual(word_to_sentence.main(["--quiet"]), 0)


class PredictedGazeLayoutTests(unittest.TestCase):
    def test_prediction_v2_matches_zero_skeleton_width(self) -> None:
        pred_cols, pred_rows = read_dicts(PRED_V2)
        zero_cols, zero_rows = read_dicts(SST_WORD_ZEROS)
        self.assertEqual(pred_cols, zero_cols)
        self.assertEqual(len(pred_rows), len(zero_rows))
        self.assertEqual(pred_rows[0]["sentence_id"], zero_rows[0]["sentence_id"])
        self.assertEqual(pred_rows[0]["word"], zero_rows[0]["word"])


if __name__ == "__main__":
    unittest.main()
