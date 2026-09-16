"""Token artifacts, join rebuild, and scaler fingerprints."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gazebook.csvio import read_dicts, table_to_array
from gazebook.paths import repo_root
from gazebook.schema import SENTENCE_RAW
from gazebook.stats import minmax, zscore
from gazebook.tokens import (
    DIGIT_TOKEN_COUNT,
    KNOWN_GLUED,
    WORDLEN_MISMATCH_COUNT,
    digit_tokens,
    glued_hits,
    load_word_averages,
    sentence_words,
    wordlen_mismatches,
)


class TokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.words = load_word_averages()

    def test_glued_catalog(self):
        hits = {h.word for h in glued_hits(self.words)}
        self.assertEqual(hits, {tok for _, tok, _ in KNOWN_GLUED})

    def test_sentence_4_empty(self):
        tokens = [w["Word"] for w in sentence_words("4_NR", self.words)]
        self.assertIn("emp11111ty", tokens)
        self.assertNotIn("empty", tokens)

    def test_sentence_80_hyphen(self):
        tokens = [w["Word"] for w in sentence_words("80_NR", self.words)]
        self.assertIn("murderoncampus", tokens)

    def test_counts(self):
        self.assertEqual(len(wordlen_mismatches(self.words)), WORDLEN_MISMATCH_COUNT)
        self.assertEqual(len(digit_tokens(self.words)), DIGIT_TOKEN_COUNT)


class JoinTests(unittest.TestCase):
    def test_standard_join(self):
        root = repo_root()
        _, text = read_dicts(root / "ZuCo_SST_data/ssts_ZuCo.csv")
        _, scaled = read_dicts(root / "ZuCo_et_csv_data/standard_scaled_average_data.csv")
        _, combined = read_dicts(root / "ZuCo_SST_data/combined_sst_et_standard.csv")
        text_by = {int(r["sentence_id"]): r for r in text}
        scaled_by = {int(r["id"]): r for r in scaled}
        for row in combined:
            sid = int(row["sentence_id"])
            self.assertEqual(text_by[sid]["sentence"], row["sentence"])
            self.assertAlmostEqual(
                float(row["nFixations"]),
                float(scaled_by[sid]["nFixations"]),
                places=12,
            )

    def test_scaler_fingerprints(self):
        root = repo_root()
        _, raw = read_dicts(root / "ZuCo_et_csv_data/average_data.csv")
        _, mm = read_dicts(root / "ZuCo_et_csv_data/min_max_scaled_average_data.csv")
        _, st = read_dicts(root / "ZuCo_et_csv_data/standard_scaled_average_data.csv")
        R = table_to_array(raw, SENTENCE_RAW)
        self.assertLess(np.max(np.abs(minmax(R) - table_to_array(mm, SENTENCE_RAW))), 1e-12)
        self.assertLess(np.max(np.abs(zscore(R, ddof=0) - table_to_array(st, SENTENCE_RAW))), 1e-12)
        # Sample std must not reconstruct the published table.
        self.assertGreater(np.max(np.abs(zscore(R, ddof=1) - table_to_array(st, SENTENCE_RAW))), 1e-4)


if __name__ == "__main__":
    unittest.main()
