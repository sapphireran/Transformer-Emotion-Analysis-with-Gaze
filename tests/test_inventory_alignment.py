import unittest

from zuco_lab.alignment import (
    feature_label_correlations,
    gaze_collinearity,
    group_words,
    label_conditioned_means,
    sent_id_from_index,
    sentence_word_profile,
)
from zuco_lab.inventory import inventory


class InventoryTests(unittest.TestCase):
    def test_inventory_covers_subjects_and_schemas(self):
        items = inventory()
        keys = {item.key for item in items}
        self.assertIn("zuco_combined_standard", keys)
        self.assertIn("subject_3", keys)
        by_key = {item.key: item for item in items}
        self.assertEqual(by_key["zuco_combined_standard"].rows, 400)
        self.assertEqual(by_key["subject_3"].rows, 299)
        self.assertTrue(by_key["subject_3"].schema_ok)
        self.assertTrue(by_key["sst_combined"].schema_ok)


class AlignmentTests(unittest.TestCase):
    def test_sentence_zero_word_profile(self):
        grouped = group_words()
        profile = sentence_word_profile("0_NR", grouped)
        self.assertEqual(profile.words[0].lower(), "presents")
        self.assertEqual(len(profile.words), 22)
        self.assertEqual(sent_id_from_index(0), "0_NR")

    def test_label_profiles_cover_all_classes(self):
        profiles = label_conditioned_means()
        self.assertEqual([p.label for p in profiles], [0, 1, 2])
        self.assertEqual(sum(p.n for p in profiles), 400)

    def test_collinearity_and_label_r(self):
        pairs = gaze_collinearity()
        self.assertGreater(len(pairs), 3)
        # Duration measures should not be independent.
        best = max(pairs, key=lambda item: abs(item[2]))
        self.assertGreater(abs(best[2]), 0.3)
        corrs = feature_label_correlations()
        self.assertEqual(set(corrs), {"nFixations", "FFD", "GPT", "TRT", "GD", "omissionRate"})


if __name__ == "__main__":
    unittest.main()
