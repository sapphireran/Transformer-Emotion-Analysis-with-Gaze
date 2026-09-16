from teag_examples.io import (
    load_full_sst_combined,
    load_full_sst_split,
    load_zuco_combined,
    load_zuco_split,
)
from teag_examples.schema import FULL_SST_SPLIT_ROWS, ZUCO_SPLIT_ROWS
from teag_examples.splits import covers_exactly, disjoint_id_sets, split_inventory


def test_zuco_split_disjoint_and_complete():
    train, valid, test = (load_zuco_split(s) for s in ("train", "valid", "test"))
    combined = load_zuco_combined()
    inv = split_inventory(train, valid, test, combined)
    assert inv["disjoint"]
    assert inv["covers_combined"]
    assert inv["n_train"] == ZUCO_SPLIT_ROWS["train"]
    assert inv["n_valid"] == ZUCO_SPLIT_ROWS["valid"]
    assert inv["n_test"] == ZUCO_SPLIT_ROWS["test"]


def test_full_sst_split_disjoint_and_complete():
    train, valid, test = (load_full_sst_split(s) for s in ("train", "valid", "test"))
    combined = load_full_sst_combined()
    assert disjoint_id_sets(train, valid, test)
    assert covers_exactly((train, valid, test), combined)
    assert len(train) == FULL_SST_SPLIT_ROWS["train"]
    assert len(valid) == FULL_SST_SPLIT_ROWS["valid"]
    assert len(test) == FULL_SST_SPLIT_ROWS["test"]
