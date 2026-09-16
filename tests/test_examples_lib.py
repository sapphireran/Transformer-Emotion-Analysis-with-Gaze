"""Unit tests for examples/lib. Stdlib unittest only."""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.lib import features, fusion, io_csv, linalg, metrics, paths, schema, stats


class StatsTests(unittest.TestCase):
    def test_mean_std(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(stats.mean(data), 2.5)
        self.assertAlmostEqual(stats.variance(data, sample=False), 1.25)
        self.assertAlmostEqual(stats.std(data, sample=False), math.sqrt(1.25))

    def test_quantile_endpoints(self) -> None:
        data = [10.0, 20.0, 30.0, 40.0]
        self.assertEqual(stats.quantile(data, 0.0), 10.0)
        self.assertEqual(stats.quantile(data, 1.0), 40.0)
        self.assertAlmostEqual(stats.quantile(data, 0.5), 25.0)

    def test_pearson_perfect(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [2.0, 4.0, 6.0, 8.0]
        self.assertAlmostEqual(stats.pearson(xs, ys), 1.0)
        self.assertAlmostEqual(stats.pearson(xs, [-y for y in ys]), -1.0)

    def test_pearson_constant_is_zero(self) -> None:
        self.assertEqual(stats.pearson([1, 1, 1], [0, 1, 2]), 0.0)

    def test_majority_baseline(self) -> None:
        winner, acc = stats.majority_baseline([0, 0, 1, 2, 0])
        self.assertEqual(winner, 0)
        self.assertAlmostEqual(acc, 0.6)

    def test_zscore_columns(self) -> None:
        matrix = [[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]]
        z = stats.zscore_columns(matrix)
        col0 = [row[0] for row in z]
        self.assertAlmostEqual(stats.mean(col0), 0.0, places=6)
        self.assertTrue(all(row[1] == 0.0 for row in z))


class LinalgTests(unittest.TestCase):
    def test_matvec(self) -> None:
        out = linalg.matvec([[1.0, 2.0], [3.0, 4.0]], [1.0, 1.0])
        self.assertEqual(out, [3.0, 7.0])

    def test_softmax_sums_to_one(self) -> None:
        probs = linalg.softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertGreater(probs[2], probs[1])
        self.assertGreater(probs[1], probs[0])

    def test_softmax_stability(self) -> None:
        probs = linalg.softmax([1000.0, 1000.0, 1001.0])
        self.assertTrue(all(math.isfinite(p) for p in probs))
        self.assertAlmostEqual(sum(probs), 1.0)

    def test_argmax_and_one_hot(self) -> None:
        self.assertEqual(linalg.argmax([0.1, 0.7, 0.2]), 1)
        self.assertEqual(linalg.one_hot(2, 4), [0.0, 0.0, 1.0, 0.0])

    def test_normalize_zero(self) -> None:
        self.assertEqual(linalg.normalize([0.0, 0.0]), [0.0, 0.0])


class MetricsTests(unittest.TestCase):
    def test_perfect_scores(self) -> None:
        y = [0, 1, 2, 0, 1, 2]
        rec = metrics.report(y, y)
        self.assertEqual(rec["accuracy"], 1.0)
        self.assertEqual(rec["macro"]["f1"], 1.0)
        self.assertEqual(rec["weighted"]["f1"], 1.0)

    def test_all_wrong(self) -> None:
        rec = metrics.report([0, 0, 0], [1, 1, 1], labels=(0, 1))
        self.assertEqual(rec["accuracy"], 0.0)
        self.assertEqual(rec["per_class"][0]["recall"], 0.0)

    def test_confusion_shape(self) -> None:
        matrix = metrics.confusion_matrix([0, 1, 1], [0, 0, 1], labels=(0, 1))
        self.assertEqual(matrix, [[1, 0], [1, 1]])


class FusionTests(unittest.TestCase):
    def test_hash_embed_is_stable(self) -> None:
        a = fusion.hash_embed("Beautifully crafted filmmaking", dim=16)
        b = fusion.hash_embed("Beautifully crafted filmmaking", dim=16)
        self.assertEqual(a, b)
        self.assertAlmostEqual(linalg.l2(a), 1.0)

    def test_tokenize_strips_punct(self) -> None:
        self.assertEqual(fusion.tokenize("Hello, world!"), ["hello", "world"])

    def test_stratified_split_keeps_all_classes(self) -> None:
        labels = [0] * 10 + [1] * 10 + [2] * 10
        items = list(range(30))
        train, test = fusion.stratified_split(items, labels, test_ratio=0.3, seed=1)
        self.assertEqual(sorted(train + test), items)
        self.assertEqual(set(train) & set(test), set())
        train_labels = {labels[i] for i in train}
        test_labels = {labels[i] for i in test}
        self.assertEqual(train_labels, {0, 1, 2})
        self.assertEqual(test_labels, {0, 1, 2})

    def test_softmax_head_fits_separable_data(self) -> None:
        rng = random.Random(0)
        features = []
        labels = []
        for i in range(60):
            label = i % 3
            vec = [0.0, 0.0, 0.0]
            vec[label] = 1.0
            vec = [v + rng.uniform(-0.05, 0.05) for v in vec]
            features.append(vec)
            labels.append(label)
        idx = list(range(60))
        result = fusion.train_softmax_head(
            "sep", features, labels, idx[:45], idx[45:], epochs=25, lr=0.4, seed=0
        )
        self.assertGreaterEqual(result.holdout_acc, 0.9)

    def test_fusion_head_forward_shapes(self) -> None:
        rng = random.Random(0)
        head = fusion.FusionHead.create(8, 5, 16, 3, rng)
        text = [0.1] * 8
        gaze = [0.2] * 5
        logits = head.logits(text, gaze)
        self.assertEqual(len(logits), 3)
        hidden = head.hidden(text, gaze)
        self.assertEqual(len(hidden), 24)

    def test_fusion_head_fits_separable_data(self) -> None:
        text, gaze, labels = [], [], []
        rng = random.Random(1)
        for i in range(72):
            label = i % 3
            t = [rng.uniform(-0.05, 0.05) for _ in range(6)]
            g = [rng.uniform(-0.05, 0.05) for _ in range(5)]
            t[label] = 1.0
            g[label] = 1.0
            text.append(t)
            gaze.append(g)
            labels.append(label)
        idx = list(range(72))
        result = fusion.train_fusion_head(
            "sep-fusion",
            text,
            gaze,
            labels,
            idx[:54],
            idx[54:],
            epochs=20,
            lr=0.08,
            gaze_out=6,
            seed=0,
        )
        self.assertTrue(all(math.isfinite(x) for x in result.losses), result.losses[-1:])
        self.assertGreaterEqual(result.holdout_acc, 0.85)

    def test_clip_vec(self) -> None:
        self.assertEqual(linalg.clip_vec([10.0, -3.0, 1.0], max_abs=2.0), [2.0, -2.0, 1.0])


class SchemaAndIoTests(unittest.TestCase):
    def test_zuco_standard_schema(self) -> None:
        path = paths.table_path("zuco_standard")
        header = io_csv.read_header(path)
        rows = io_csv.read_rows(path)
        report = schema.validate_rows("zuco_standard", header, rows)
        self.assertTrue(report.ok, [i.message for i in report.issues])
        self.assertEqual(len(rows), 400)

    def test_sst_train_schema_and_aliases(self) -> None:
        path = paths.table_path("sst_train")
        header = io_csv.read_header(path)
        rows = io_csv.read_rows(path)
        report = schema.validate_rows("sst_train", header, rows)
        self.assertTrue(report.ok, [i.message for i in report.issues])
        self.assertEqual(len(rows), 9482)
        matrix = features.matrix_for(rows[:3], "sst5")
        self.assertEqual(len(matrix[0]), 5)

    def test_missing_column_is_error(self) -> None:
        report = schema.validate_rows(
            "zuco_text",
            ["sentence_id", "sentence"],
            [{"sentence_id": "0", "sentence": "x"}],
        )
        self.assertFalse(report.ok)

    def test_bad_label_is_error(self) -> None:
        report = schema.validate_rows(
            "zuco_text",
            ["sentence_id", "sentence", "sentiment_label"],
            [{"sentence_id": "0", "sentence": "x", "sentiment_label": "9"}],
        )
        self.assertFalse(report.ok)


class PathsTests(unittest.TestCase):
    def test_every_named_table_exists(self) -> None:
        missing = [name for name in paths.TABLES if not paths.table_path(name).is_file()]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
