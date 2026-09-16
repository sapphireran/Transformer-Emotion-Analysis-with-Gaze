from tea_gaze.io import inventory, load_and_validate, load_rows
from tea_gaze.paths import DATASETS, repo_root, subject_et_paths
from tea_gaze.schema import summarize_labels, validate_sentence_rows


def test_repo_contains_core_personal_csvs():
    for key in (
        "zuco_sst_standard",
        "zuco_sst_labels",
        "zuco_et_average",
        "zuco_word_averages",
        "full_sst_train",
    ):
        assert DATASETS[key].exists(), key


def test_zuco_sst_standard_validates():
    rows, report = load_and_validate("zuco_sst_standard")
    assert report.ok, report.issues
    assert len(rows) == 400
    counts = summarize_labels(rows)
    assert counts[0] + counts[1] + counts[2] == 400
    assert counts[2] == 140
    assert counts[1] == 137
    assert counts[0] == 123


def test_full_sst_uses_nfix_alias():
    rows = load_rows("full_sst_train")
    assert "nFix" in rows[0]
    assert "nFixations" not in rows[0]
    report = validate_sentence_rows("full_sst_train", rows[:25])
    assert report.ok, report.issues


def test_subject_three_is_shorter():
    from tea_gaze.io import _open_csv

    paths = subject_et_paths()
    assert len(paths) == 12
    assert all(path.is_file() for path in paths)
    assert len(_open_csv(paths[0])) == 400
    assert len(_open_csv(paths[2])) == 299


def test_inventory_lists_checked_in_files():
    items = inventory()
    assert len(items) == len(DATASETS)
    present = [item for item in items if item.exists]
    assert len(present) == len(items)
    zuco = next(item for item in items if item.key == "zuco_sst_standard")
    assert zuco.rows == 400
    assert zuco.gaze_source == "measured"


def test_repo_root_points_at_workspace():
    root = repo_root()
    assert (root / "model_ZuCo_SST.py").is_file()
    assert (root / "ZuCo_SST_data").is_dir()
