import unittest

from zuco_lab import numbers


class NumbersTests(unittest.TestCase):
    def test_mean_pstdev_zscore(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(numbers.mean(values), 3.0)
        self.assertAlmostEqual(numbers.pstdev(values), 2**0.5)
        z = numbers.zscore(values)
        self.assertAlmostEqual(numbers.mean(z), 0.0, places=7)
        self.assertAlmostEqual(numbers.pstdev(z), 1.0, places=7)

    def test_pearson_perfect_and_uncorrelated(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        self.assertAlmostEqual(numbers.pearson(xs, xs), 1.0)
        self.assertAlmostEqual(numbers.pearson(xs, [3.0, 2.0, 1.0, 0.0]), -1.0)
        self.assertEqual(numbers.pearson([1.0, 1.0, 1.0], [0.0, 1.0, 2.0]), 0.0)

    def test_spearman_monotonic(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [1.0, 4.0, 9.0, 16.0]
        self.assertAlmostEqual(numbers.spearman(xs, ys), 1.0)

    def test_softmax_and_argmax(self):
        probs = numbers.softmax([0.0, 0.0, 0.0])
        self.assertTrue(all(abs(p - 1 / 3) < 1e-9 for p in probs))
        self.assertEqual(numbers.argmax([0.1, 0.7, 0.2]), 1)

    def test_metrics(self):
        labels = [0, 0, 1, 2]
        preds = [0, 1, 1, 2]
        self.assertAlmostEqual(numbers.accuracy(preds, labels), 0.75)
        self.assertGreater(numbers.weighted_f1(preds, labels), 0.0)
        self.assertEqual(numbers.majority(labels), 0)

    def test_matvec_concat(self):
        out = numbers.matvec([[1.0, 2.0], [3.0, 4.0]], [1.0, 1.0])
        self.assertEqual(out, [3.0, 7.0])
        self.assertEqual(numbers.concat([1], [2, 3]), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
