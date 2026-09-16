import unittest

from zuco_lab.subjects import (
    SUBJECT3_COMPACT_ROWS,
    compact_to_original_subject3,
    contamination_table,
    first_mismatch_id,
    original_to_compact_subject3,
    reader_agreement,
    remap_word_sent_id,
    remapped_sentlen_matches,
)


class SubjectAlignmentTests(unittest.TestCase):
    def test_compact_mapping(self):
        self.assertEqual(compact_to_original_subject3(0), 0)
        self.assertEqual(compact_to_original_subject3(149), 149)
        self.assertEqual(compact_to_original_subject3(150), 250)
        self.assertEqual(compact_to_original_subject3(298), 398)
        self.assertEqual(SUBJECT3_COMPACT_ROWS, 299)
        self.assertIsNone(original_to_compact_subject3(200))
        self.assertEqual(original_to_compact_subject3(250), 150)

    def test_first_sentlen_mismatch_is_150(self):
        self.assertEqual(first_mismatch_id(), 150)

    def test_remapped_sentlen_matches_subject_one(self):
        self.assertTrue(remapped_sentlen_matches())

    def test_contamination_zero_on_aligned_window(self):
        rows = contamination_table("nFixations", ids=range(0, 20))
        self.assertTrue(all(abs(row.delta) < 1e-12 for row in rows))
        self.assertTrue(all(row.n_index == 12 and row.n_aligned == 12 for row in rows))

    def test_contamination_nonzero_after_skip(self):
        rows = contamination_table("SentLen", ids=range(150, 155))
        self.assertTrue(any(abs(row.delta) > 0.1 for row in rows))
        # Index mean includes the remapped reader-3 sentence; aligned mean does not.
        self.assertTrue(all(row.n_index == 12 for row in rows))
        self.assertTrue(all(row.n_aligned == 11 for row in rows))

    def test_reader_agreement_reasonable(self):
        report = reader_agreement("nFixations", 0, 40)
        self.assertGreater(report.mean_pairwise_r, 0.05)
        self.assertLess(report.mean_pairwise_r, 0.95)
        self.assertEqual(report.n_pairs, 66)

    def test_word_sent_id_remap(self):
        self.assertEqual(remap_word_sent_id(1, "150_NR"), "150_NR")
        self.assertEqual(remap_word_sent_id(3, "150_NR"), "250_NR")
        self.assertEqual(remap_word_sent_id(3, "0_NR"), "0_NR")


if __name__ == "__main__":
    unittest.main()
