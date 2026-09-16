from examples.lib.paths import (
    REPO_ROOT,
    expected_subject_sentence_rows,
    expected_subject_word_rows,
    iter_dataset_specs,
    spec_by_key,
    subject_sentence_csv,
    subject_word_csv,
)


def test_repo_root_has_trainers():
    assert (REPO_ROOT / "model_ZuCo_SST.py").is_file()
    assert (REPO_ROOT / "model_full_SST.py").is_file()


def test_every_inventory_file_exists():
    missing = [spec.relative for spec in iter_dataset_specs() if not spec.exists()]
    assert missing == []


def test_spec_by_key_roundtrip():
    spec = spec_by_key("zuco_standard")
    assert spec.expected_rows == 400
    assert spec.gaze_kind == "measured"


def test_spec_by_key_unknown():
    try:
        spec_by_key("not-a-dataset")
    except KeyError as exc:
        assert "not-a-dataset" in str(exc)
    else:
        raise AssertionError("expected KeyError")


def test_subject_expected_lengths():
    assert expected_subject_sentence_rows(1) == 400
    assert expected_subject_sentence_rows(3) == 299
    assert expected_subject_word_rows(3) == 5293
    assert expected_subject_word_rows(12) == 7129


def test_subject_paths():
    assert subject_sentence_csv(1).name == "1_SR.csv"
    assert subject_word_csv(12).name == "12_SR.csv"
    try:
        subject_sentence_csv(0)
    except ValueError:
        pass
    else:
        raise AssertionError("subject 0 should fail")
