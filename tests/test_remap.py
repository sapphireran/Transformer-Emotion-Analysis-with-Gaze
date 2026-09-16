"""Lock reader-3 compaction and the published positional average."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gazebook.remap import (
    NFIX_CONTAMINATED,
    READER3_COMPACT_ROWS,
    SENTLEN_CONTAMINATED,
    WORD_ALIGN_ROWS,
    WORD_MISMATCH_IN_OVERLAP,
    WORST_NFIX_ID,
    compact_to_original,
    contamination_report,
    load_subject_tables,
    original_to_compact,
    positional_average,
    published_column,
    word_align_row_count,
    word_mismatch_count,
)


class RemapUnitTests(unittest.TestCase):
    def test_round_trip_kept_ids(self):
        for orig in list(range(150)) + list(range(250, 399)):
            compact = original_to_compact(orig)
            self.assertIsNotNone(compact)
            self.assertEqual(compact_to_original(compact), orig)

    def test_dropped_ids(self):
        for orig in list(range(150, 250)) + [399]:
            self.assertIsNone(original_to_compact(orig))

    def test_cut_examples(self):
        self.assertEqual(compact_to_original(149), 149)
        self.assertEqual(compact_to_original(150), 250)
        self.assertEqual(compact_to_original(298), 398)
        self.assertEqual(original_to_compact(250), 150)


class RemapDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = load_subject_tables()

    def test_reader3_length(self):
        self.assertEqual(len(self.tables[2]), READER3_COMPACT_ROWS)

    def test_sentlen_smoking_gun(self):
        # Compact 150 is a 15-word sentence; published sentence 150 is 9 words.
        self.assertEqual(float(self.tables[0][150]["SentLen"]), 9.0)
        self.assertEqual(float(self.tables[2][150]["SentLen"]), 15.0)
        pub = published_column(col="SentLen")
        self.assertAlmostEqual(pub[150], (9.0 * 11 + 15.0) / 12, places=10)

    def test_positional_rebuild(self):
        pub = published_column(col="nFixations")
        rebuilt = positional_average(self.tables, "nFixations")
        self.assertLess(abs(rebuilt - pub).max(), 1e-12)

    def test_nfix_contamination(self):
        pub = published_column(col="nFixations")
        report = contamination_report(self.tables, pub, "nFixations")
        self.assertEqual(report.n_changed, NFIX_CONTAMINATED)
        self.assertEqual(report.worst_id, WORST_NFIX_ID)
        self.assertGreater(report.max_abs, 0.2)

    def test_sentlen_contamination(self):
        pub = published_column(col="SentLen")
        report = contamination_report(self.tables, pub, "SentLen")
        self.assertEqual(report.n_changed, SENTLEN_CONTAMINATED)

    def test_word_stream(self):
        self.assertEqual(word_align_row_count(), WORD_ALIGN_ROWS)
        self.assertEqual(word_mismatch_count(), WORD_MISMATCH_IN_OVERLAP)


if __name__ == "__main__":
    unittest.main()
