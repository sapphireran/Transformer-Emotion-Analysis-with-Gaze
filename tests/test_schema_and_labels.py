import unittest

from zuco_lab import csvio, labels, paths
from zuco_lab.schema import (
    SST_SENTENCE_GAZE,
    SUBJECT_SENTENCE,
    ZUCO_SENTENCE_GAZE,
    SchemaError,
    validate_rows,
)


class SchemaLabelTests(unittest.TestCase):
    def test_zuco_combined_validates(self):
        rows = csvio.read_dicts(paths.ZUCO_COMBINED_STANDARD)
        validate_rows(rows, ZUCO_SENTENCE_GAZE)
        self.assertEqual(len(rows), 400)
        self.assertEqual(labels.label_counts(rows), {0: 123, 1: 137, 2: 140})

    def test_sst_combined_validates(self):
        rows = csvio.read_dicts(paths.SST_COMBINED)
        validate_rows(rows, SST_SENTENCE_GAZE)
        self.assertEqual(len(rows), 11853)
        counts = labels.label_counts(rows)
        self.assertEqual(counts[0], 4649)
        self.assertEqual(counts[1], 2241)
        self.assertEqual(counts[2], 4963)

    def test_subject_one_validates(self):
        rows = csvio.read_dicts(paths.subject_sentence_csv(1))
        validate_rows(rows, SUBJECT_SENTENCE)
        self.assertEqual(len(rows), 400)

    def test_missing_column_raises(self):
        rows = [{"sentence_id": "0", "sentence": "x", "sentiment_label": "1"}]
        with self.assertRaises(SchemaError):
            validate_rows(rows, ZUCO_SENTENCE_GAZE)

    def test_label_helpers(self):
        self.assertEqual(labels.parse_label("POSITIVE"), 2)
        self.assertEqual(labels.label_name(0), "negative")
        rates = labels.label_rates({0: 1, 1: 1, 2: 2})
        self.assertAlmostEqual(rates[2], 0.5)


if __name__ == "__main__":
    unittest.main()
