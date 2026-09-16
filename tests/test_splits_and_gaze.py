"""Split integrity, skip rates, and metric helpers."""

from __future__ import annotations

import unittest

from examples.lib.gaze import (
    FULL_SST_GAZE,
    feature_label_anova,
    inter_subject_cv,
    per_class_means,
    sentence_skip_rate,
    skip_rate_by_word_length,
    subject_feature_panel,
)
from examples.lib.loading import (
    documented_datasets,
    load_dataset,
    load_full_sst_splits,
    load_subject_sentence_tables,
    load_zuco_word_average,
)
from examples.lib.metrics import majority_baseline, weighted_classification_scores
from examples.lib.paths import repo_root


class SplitIntegrityTest(unittest.TestCase):
    def test_full_sst_splits_are_disjoint_and_cover_combined(self) -> None:
        root = repo_root()
        splits = load_full_sst_splits(root=root)
        combined = load_dataset(
            next(s for s in documented_datasets() if s.key == "sst_combined"), root=root
        )
        ids = {name: set(df["sentence_id"]) for name, df in splits.items()}
        self.assertFalse(ids["train"] & ids["valid"])
        self.assertFalse(ids["train"] & ids["test"])
        self.assertFalse(ids["valid"] & ids["test"])
        self.assertEqual(ids["train"] | ids["valid"] | ids["test"], set(combined["sentence_id"]))
        self.assertEqual(sum(len(df) for df in splits.values()), len(combined))


class GazeStatsTest(unittest.TestCase):
    def test_per_class_means_have_three_rows(self) -> None:
        sst = load_dataset(
            next(s for s in documented_datasets() if s.key == "sst_combined"),
            root=repo_root(),
        )
        means = per_class_means(sst, FULL_SST_GAZE)
        self.assertEqual(set(means.index), {"negative", "neutral", "positive"})
        self.assertEqual(list(means.columns), list(FULL_SST_GAZE))

    def test_anova_returns_one_row_per_feature(self) -> None:
        sst = load_dataset(
            next(s for s in documented_datasets() if s.key == "sst_combined"),
            root=repo_root(),
        )
        table = feature_label_anova(sst, FULL_SST_GAZE)
        self.assertEqual(len(table), len(FULL_SST_GAZE))
        self.assertTrue((table["p_value"] >= 0).all())
        self.assertTrue((table["p_value"] <= 1).all())

    def test_skip_rates_are_probabilities(self) -> None:
        words = load_zuco_word_average(root=repo_root())
        by_len = skip_rate_by_word_length(words)
        self.assertTrue(((by_len["skip_rate"] >= 0) & (by_len["skip_rate"] <= 1)).all())
        sent = sentence_skip_rate(words)
        self.assertTrue(((sent["skip_rate"] >= 0) & (sent["skip_rate"] <= 1)).all())
        overall = float(words["nFixations"].fillna(0).eq(0).mean())
        self.assertGreater(overall, 0.0)
        self.assertLess(overall, 1.0)

    def test_inter_subject_cv_is_finite_for_core_features(self) -> None:
        panel = subject_feature_panel(load_subject_sentence_tables(root=repo_root()))
        cv = inter_subject_cv(panel, ("nFixations", "TRT", "FFD"))
        core = cv.dropna(subset=["subject_cv"])
        self.assertGreater(len(core), 0)
        self.assertTrue((core["subject_cv"] >= 0).all())
        self.assertEqual(set(cv["feature"]), {"nFixations", "TRT", "FFD"})


class MetricsTest(unittest.TestCase):
    def test_perfect_predictions_score_one(self) -> None:
        labels = [0, 1, 2, 2]
        scores = weighted_classification_scores(labels, labels)
        self.assertAlmostEqual(scores["accuracy"], 1.0)
        self.assertAlmostEqual(scores["f1_weighted"], 1.0)

    def test_majority_baseline_matches_largest_class_share(self) -> None:
        labels = [0, 0, 0, 1, 2]
        scores = majority_baseline(labels)
        self.assertEqual(scores["majority_class"], 0)
        self.assertAlmostEqual(scores["accuracy"], 0.6)


if __name__ == "__main__":
    unittest.main()
