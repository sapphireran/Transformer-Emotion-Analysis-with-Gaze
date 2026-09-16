"""CSV loading and documented row counts."""

from __future__ import annotations

import unittest

from examples.lib.loading import (
    LABEL_NAMES,
    SUBJECT_SENTENCE_ROWS,
    documented_datasets,
    load_dataset,
    load_sst_raw,
    load_subject_sentence_tables,
    load_zuco_combined,
    load_zuco_word_average,
    remap_subject3_original_ids,
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

    def test_twelve_subject_tables_have_known_row_counts(self) -> None:
        tables = load_subject_sentence_tables(root=repo_root())
        self.assertEqual(len(tables), 12)
        for subject, df in tables.items():
            with self.subTest(subject=subject):
                self.assertEqual(len(df), SUBJECT_SENTENCE_ROWS[subject])

    def test_subject3_remap_restores_sentlen_alignment(self) -> None:
        tables = load_subject_sentence_tables(root=repo_root())
        ref = tables[1]
        remapped = remap_subject3_original_ids(tables[3])
        self.assertEqual(remapped["id"].tolist(), list(range(150)) + list(range(250, 399)))
        merged = ref.merge(remapped, on="id", suffixes=("_s1", "_s3"))
        self.assertEqual(len(merged), 299)
        self.assertTrue((merged["SentLen_s1"] == merged["SentLen_s3"]).all())
        naive = ref.merge(tables[3], on="id", suffixes=("_s1", "_s3"))
        late = naive.loc[naive["id"] >= 150]
        self.assertLess((late["SentLen_s1"] == late["SentLen_s3"]).mean(), 0.05)

    def test_word_average_sent_id_prefix_covers_zuco_sentences(self) -> None:
        words = load_zuco_word_average(root=repo_root())
        zuco = load_zuco_combined(root=repo_root())
        prefixes = set(
            words["Sent_ID"].astype(str).str.split("_").str[0].astype(int).tolist()
        )
        self.assertTrue(set(zuco["sentence_id"]).issubset(prefixes))


if __name__ == "__main__":
    unittest.main()
