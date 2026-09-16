"""Gaze-only baselines, permutation ANOVA, and length residualization."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from examples.lib.baselines import (
    holdout_scores,
    permute_anova,
    residualize_on_length,
    stratified_cv_scores,
)
from examples.lib.gaze import FULL_SST_GAZE, ZUCO_FUSION_GAZE, words_for_sentence
from examples.lib.loading import load_full_sst_splits, load_zuco_combined, load_zuco_word_average
from examples.lib.paths import repo_root


class GazeOnlyBaselineTest(unittest.TestCase):
    def test_zuco_cv_scores_are_probabilities(self) -> None:
        zuco = load_zuco_combined(root=repo_root())
        scores = stratified_cv_scores(
            zuco[list(ZUCO_FUSION_GAZE)], zuco["sentiment_label"], n_splits=5, seed=0
        )
        self.assertGreaterEqual(scores["accuracy"], 0.0)
        self.assertLessEqual(scores["accuracy"], 1.0)
        self.assertEqual(scores["n"], 400)
        # Gaze-only should not be a perfect classifier.
        self.assertLess(scores["accuracy"], 0.9)

    def test_full_sst_holdout_runs(self) -> None:
        splits = load_full_sst_splits(root=repo_root())
        scores = holdout_scores(
            splits["train"][list(FULL_SST_GAZE)],
            splits["train"]["sentiment_label"],
            splits["valid"][list(FULL_SST_GAZE)],
            splits["valid"]["sentiment_label"],
            seed=0,
        )
        self.assertEqual(scores["n_train"], 9482)
        self.assertEqual(scores["n_test"], 1185)
        self.assertGreaterEqual(scores["accuracy"], 0.0)
        self.assertLessEqual(scores["accuracy"], 1.0)


class PermuteAnovaTest(unittest.TestCase):
    def test_shuffled_labels_have_small_typical_f(self) -> None:
        zuco = load_zuco_combined(root=repo_root())
        table = permute_anova(zuco, ZUCO_FUSION_GAZE, n_perm=20, seed=0)
        self.assertEqual(len(table), len(ZUCO_FUSION_GAZE))
        self.assertTrue(((table["perm_p"] >= 0) & (table["perm_p"] <= 1)).all())
        self.assertTrue((table["perm_mean_f"] >= 0).all())


class LengthResidualTest(unittest.TestCase):
    def test_residualize_uncorrelates_linear_length(self) -> None:
        rng = np.random.default_rng(0)
        length = pd.Series(np.linspace(5, 40, 80))
        df = pd.DataFrame(
            {
                "feat": 2.0 * length + rng.normal(scale=0.01, size=80),
                "sentiment_label": np.repeat([0, 1, 2], repeats=[30, 25, 25]),
            }
        )
        residual = residualize_on_length(df, ["feat"], length)
        r = float(pd.Series(residual["feat"]).corr(length))
        self.assertLess(abs(r), 0.05)


class SentenceWalkthroughTest(unittest.TestCase):
    def test_default_sentence_ids_have_words(self) -> None:
        zuco = load_zuco_combined(root=repo_root())
        words = load_zuco_word_average(root=repo_root())
        for sentence_id in (0, 3, 4, 316):
            with self.subTest(sentence_id=sentence_id):
                self.assertIn(sentence_id, set(zuco["sentence_id"]))
                tokens = words_for_sentence(words, sentence_id)
                self.assertGreater(len(tokens), 0)

    def test_known_word_token_corruption_still_present(self) -> None:
        words = load_zuco_word_average(root=repo_root())
        tokens = words_for_sentence(words, 4)
        self.assertIn("emp11111ty", set(tokens["Word"].astype(str)))


if __name__ == "__main__":
    unittest.main()
