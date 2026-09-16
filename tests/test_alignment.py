"""Subject-3 packing map, proven against SentLen and word tokens in the CSVs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from sidecar.alignment import (  # noqa: E402
    N_PACKED_SENTENCES,
    TASK1_SKIPPED_ORIGINALS,
    pack_subject3_id,
    subject3_sentence_alignment,
    unpack_subject3_id,
    word_break_row,
)
from sidecar.load import load_subject_sentence, load_subject_word  # noqa: E402


class PackingMapTests(unittest.TestCase):
    def test_round_trip_kept_ids(self) -> None:
        for original in range(400):
            packed = pack_subject3_id(original)
            if original in TASK1_SKIPPED_ORIGINALS:
                self.assertIsNone(packed)
            else:
                self.assertEqual(unpack_subject3_id(packed), original)

    def test_known_landmarks(self) -> None:
        self.assertEqual(unpack_subject3_id(0), 0)
        self.assertEqual(unpack_subject3_id(149), 149)
        self.assertEqual(unpack_subject3_id(150), 250)
        self.assertEqual(unpack_subject3_id(298), 398)
        self.assertIsNone(pack_subject3_id(150))
        self.assertIsNone(pack_subject3_id(399))
        self.assertEqual(pack_subject3_id(250), 150)
        self.assertEqual(len(TASK1_SKIPPED_ORIGINALS), 101)
        self.assertEqual(N_PACKED_SENTENCES, 299)

    def test_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            unpack_subject3_id(299)
        with self.assertRaises(ValueError):
            pack_subject3_id(400)


class AlignmentAgainstCsvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.s1 = load_subject_sentence(1)
        cls.s3 = load_subject_sentence(3)
        cls.w1 = load_subject_word(1)
        cls.w3 = load_subject_word(3)

    def test_subject3_is_299_rows(self) -> None:
        self.assertEqual(len(self.s3), 299)
        self.assertEqual(int(self.s3["id"].max()), 298)
        self.assertEqual(len(self.s1), 400)

    def test_unpacked_sentlen_matches_everywhere(self) -> None:
        align = subject3_sentence_alignment(self.s1, self.s3)
        self.assertTrue(bool(align["sentlen_match"].all()))
        # Naive same-id pairing must fail at packed 150.
        row150 = align.loc[align["packed_id"] == 150].iloc[0]
        self.assertFalse(bool(row150["same_row_would_match"]))
        self.assertEqual(int(row150["original_id"]), 250)

    def test_word_break_and_token_identity(self) -> None:
        self.assertEqual(word_break_row(self.w1), 2594)
        w1_sid = self.w1["Sent_ID"].astype(str).str.split("_").str[0].astype(int)
        w3_sid = self.w3["Sent_ID"].astype(str).str.split("_").str[0].astype(int)
        prefix = (w1_sid <= 149).sum()
        self.assertEqual(int(prefix), 2594)
        w1_prefix = self.w1.loc[w1_sid <= 149, "Word"].fillna("").tolist()
        w3_prefix = self.w3.loc[w3_sid <= 149, "Word"].fillna("").tolist()
        self.assertEqual(w1_prefix, w3_prefix)
        w3_150 = self.w3.loc[w3_sid == 150, "Word"].fillna("").tolist()
        w1_250 = self.w1.loc[w1_sid == 250, "Word"].fillna("").tolist()
        w1_150 = self.w1.loc[w1_sid == 150, "Word"].fillna("").tolist()
        self.assertEqual(w3_150, w1_250)
        self.assertNotEqual(w3_150, w1_150)


if __name__ == "__main__":
    unittest.main()
