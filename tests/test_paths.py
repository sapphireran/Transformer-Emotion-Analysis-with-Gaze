import pandas as pd
import pytest

from gazekit.paths import default_paths, repo_root
from gazekit.schema import EXPECTED_ROWS


def test_repo_root_contains_training_scripts():
    root = repo_root()
    assert (root / "model_ZuCo_SST.py").is_file()
    assert (root / "model_full_SST.py").is_file()
    assert (root / "utils_ZuCo.py").is_file()


def test_default_paths_point_at_real_files():
    paths = default_paths()
    for path in (
        paths.zuco_combined_standard,
        paths.zuco_combined_minmax,
        paths.zuco_text,
        paths.zuco_train,
        paths.word_averages,
        paths.full_sst_train,
        paths.predicted_gaze_v2,
    ):
        assert path.is_file(), path


def test_subject_paths_and_row_counts():
    paths = default_paths()
    with pytest.raises(ValueError):
        paths.subject_sentence_et(0)
    with pytest.raises(ValueError):
        paths.subject_sentence_et(13)
    for i in range(1, 13):
        csv = paths.subject_sentence_et(i)
        assert csv.is_file(), csv
        n = len(pd.read_csv(csv))
        if i == 3:
            assert n == EXPECTED_ROWS["zuco_subject_3"]
        else:
            assert n == EXPECTED_ROWS["zuco_subject_default"]
