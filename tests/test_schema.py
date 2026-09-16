"""Lock the committed table shapes and label mixes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gazebook.schema import (
    FULL_SST_LAST_BATCH,
    FULL_SST_TEST_ROWS,
    SUBJECT_ROW_COUNTS,
    WORD_ROW_COUNTS,
    check_label_mix,
    check_subject_lengths,
    check_tables,
)


class SchemaTests(unittest.TestCase):
    def test_tables(self):
        result = check_tables()
        self.assertTrue(result.ok, result.errors)

    def test_labels(self):
        result = check_label_mix()
        self.assertTrue(result.ok, result.errors)

    def test_subjects(self):
        result = check_subject_lengths()
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(SUBJECT_ROW_COUNTS[3], 299)
        self.assertEqual(WORD_ROW_COUNTS[3], 5293)
        self.assertEqual(SUBJECT_ROW_COUNTS[1], 400)

    def test_last_batch(self):
        self.assertEqual(FULL_SST_TEST_ROWS, 1186)
        self.assertEqual(FULL_SST_LAST_BATCH, 162)
        self.assertEqual(1186 % 256, 162)


if __name__ == "__main__":
    unittest.main()
