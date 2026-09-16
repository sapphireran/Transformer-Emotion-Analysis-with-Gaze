"""Inventory, split disjointness, and sidecar PCA smoke tests on committed CSVs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from sidecar.load import (  # noqa: E402
    load_sst_combined,
    load_sst_test,
    load_sst_train,
    load_sst_valid,
    load_zuco_standard,
)
from sidecar.paths import SST_GAZE_COLS, expected_tables  # noqa: E402
from sidecar.rank import gaze_pca  # noqa: E402


class InventoryTests(unittest.TestCase):
    def test_every_expected_table_exists(self) -> None:
        missing = [name for name, path in expected_tables().items() if not path.is_file()]
        self.assertEqual(missing, [])


class SplitTests(unittest.TestCase):
    def test_full_sst_ids_partition_combined(self) -> None:
        train, valid, test, comb = (
            load_sst_train(),
            load_sst_valid(),
            load_sst_test(),
            load_sst_combined(),
        )
        ids = set(train.sentence_id) | set(valid.sentence_id) | set(test.sentence_id)
        self.assertEqual(ids, set(comb.sentence_id))
        self.assertEqual(len(set(train.sentence_id) & set(valid.sentence_id)), 0)
        self.assertEqual(len(set(train.sentence_id) & set(test.sentence_id)), 0)
        self.assertEqual(len(set(valid.sentence_id) & set(test.sentence_id)), 0)
        self.assertEqual(len(train) + len(valid) + len(test), len(comb))

    def test_known_text_leaks(self) -> None:
        train, valid, test = load_sst_train(), load_sst_valid(), load_sst_test()
        self.assertEqual(len(set(train.sentence) & set(valid.sentence)), 1)
        self.assertEqual(len(set(train.sentence) & set(test.sentence)), 1)
        self.assertEqual(len(set(valid.sentence) & set(test.sentence)), 0)

    def test_zuco_and_full_sst_are_almost_disjoint(self) -> None:
        zuco = load_zuco_standard()
        comb = load_sst_combined()
        self.assertEqual(len(set(zuco.sentence) & set(comb.sentence)), 3)
        self.assertEqual(len(zuco), 400)
        self.assertEqual(len(comb), 11853)


class PcaTests(unittest.TestCase):
    def test_first_component_dominates_full_sst_gaze(self) -> None:
        pca = gaze_pca(load_sst_train(), SST_GAZE_COLS)
        explained = pca["explained_variance_ratio"]
        self.assertGreater(float(explained[0]), 0.90)
        self.assertGreater(float(explained[1]), 0.07)
        self.assertLess(float(explained[2:].sum()), 0.01)


if __name__ == "__main__":
    unittest.main()
