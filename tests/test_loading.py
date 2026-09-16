"""CSV loading and documented row counts."""

from __future__ import annotations

import unittest

from examples.lib.loading import (
    LABEL_NAMES,
    documented_datasets,
    load_dataset,
    load_sst_raw,
    load_subject_sentence_tables,
    load_zuco_combined,
    load_zuco_word_average,
)
from examples.lib.paths import repo_root


class DocumentedDatasetsTest(unittest.TestCase):
    def test_every_spec_loads_and_matches_row_count(self) -> None:
        for spec in documented_datasets():
            with self.subTest(spec.key):
                df = load_dataset(spec, root=repo_root())
                self.assertEqual(len(df), spec.expected_rows)
                for column in spec.required_columns:
                    self.assertIn(column, df.columns)

    def test_raw_sst_has_string_labels_and_11853_rows(self) -> None:
        df = load_sst_raw(root=repo_root())
        self.assertEqual(len(df), 11853)
        self.assertEqual(set(df["sentiment"].unique()), {"NEGATIVE", "NEUTRAL", "POSITIVE"})

    def test_zuco_labels_are_0_1_2(self) -> None:
        df = load_zuco_combined(root=repo_root())
        self.assertEqual(set(df["sentiment_label"].unique()), {0, 1, 2})
        self.assertEqual(set(LABEL_NAMES), {0, 1, 2})

    def test_twelve_subject_tables_align_on_id(self) -> None:
        tables = load_subject_sentence_tables(root=repo_root())
        self.assertEqual(len(tables), 12)
        reference = tables[1]["id"].tolist()
        for subject, df in tables.items():
            with self.subTest(subject=subject):
                self.assertEqual(len(df), 400)
                self.assertEqual(df["id"].tolist(), reference)

    def test_word_average_sent_id_prefix_covers_zuco_sentences(self) -> None:
        words = load_zuco_word_average(root=repo_root())
        zuco = load_zuco_combined(root=repo_root())
        prefixes = set(
            words["Sent_ID"].astype(str).str.split("_").str[0].astype(int).tolist()
        )
        self.assertTrue(set(zuco["sentence_id"]).issubset(prefixes))


if __name__ == "__main__":
    unittest.main()
