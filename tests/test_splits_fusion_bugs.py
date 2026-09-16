import unittest

from zuco_lab.baselines import evaluate_majority, fit_gaze_ovr, stratified_holdout
from zuco_lab.bugs import simulate_test_loop
from zuco_lab.fusion import LateFusionToy, tokenize
from zuco_lab.splits import audit_sst, audit_zuco, text_overlap


class SplitTests(unittest.TestCase):
    def test_zuco_partitions(self):
        audit = audit_zuco()
        self.assertEqual((audit.n_train, audit.n_valid, audit.n_test, audit.n_combined), (320, 40, 40, 400))
        self.assertEqual(audit.train_valid_overlap, 0)
        self.assertEqual(audit.train_test_overlap, 0)
        self.assertEqual(audit.valid_test_overlap, 0)
        self.assertEqual(audit.missing_from_splits, 0)
        self.assertEqual(audit.valid_labels[0], 7)
        self.assertEqual(audit.valid_labels[2], 19)

    def test_sst_partitions(self):
        audit = audit_sst()
        self.assertEqual(audit.n_combined, 11853)
        self.assertEqual(audit.n_train + audit.n_valid + audit.n_test, 11853)
        self.assertEqual(audit.train_valid_overlap, 0)
        self.assertEqual(audit.missing_from_splits, 0)

    def test_text_overlap_is_three_and_labels_agree(self):
        shared = text_overlap()
        self.assertEqual(len(shared), 3)
        self.assertTrue(all(z == s for _, z, s in shared))


class FusionTests(unittest.TestCase):
    def test_seeded_forward_is_stable(self):
        model = LateFusionToy.seeded(7)
        tokens = tokenize("Slow, silly and unintentionally hilarious.")
        gaze = [0.1, -0.2, 0.0, 0.3, -0.1]
        pred_a, probs_a, _ = model.predict(tokens, gaze)
        pred_b, probs_b, _ = LateFusionToy.seeded(7).predict(tokens, gaze)
        self.assertEqual(pred_a, pred_b)
        self.assertEqual(len(probs_a), 3)
        self.assertAlmostEqual(sum(probs_a), 1.0, places=6)
        self.assertEqual(probs_a, probs_b)

    def test_gaze_dimension_guard(self):
        model = LateFusionToy.seeded(1)
        with self.assertRaises(ValueError):
            model.encode_gaze([0.0, 0.0])


class BugTests(unittest.TestCase):
    def test_overwrite_covers_last_batch_only(self):
        demo = simulate_test_loop(1186, 256)
        self.assertEqual(demo.n_batches, 5)
        self.assertEqual(demo.overwrite_covers, 1186 - 256 * 4)
        self.assertEqual(len(demo.extend_preds), 1186)
        self.assertGreater(demo.extend_accuracy, demo.overwrite_accuracy)


class BaselineTests(unittest.TestCase):
    def test_majority_and_holdout(self):
        labels = [0] * 6 + [1] * 6 + [2] * 6
        report = evaluate_majority(labels)
        self.assertAlmostEqual(report.accuracy, 1 / 3)
        train, test = stratified_holdout(labels, frac=0.5, seed=0)
        self.assertEqual(len(train) + len(test), 18)

    def test_gaze_ovr_separates_easy_cloud(self):
        # Class k lives near basis vector k (padded to 5-d).
        x, y = [], []
        for k in range(3):
            for _ in range(8):
                row = [0.05] * 5
                row[k] = 1.0
                x.append(row)
                y.append(k)
        model = fit_gaze_ovr(x, y)
        preds = [model.predict(row) for row in x]
        self.assertGreaterEqual(sum(p == t for p, t in zip(preds, y)) / len(y), 0.9)


if __name__ == "__main__":
    unittest.main()
