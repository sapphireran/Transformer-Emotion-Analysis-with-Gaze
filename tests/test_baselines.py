import pytest

from tea_gaze.io import load_frame


pytest.importorskip("sklearn")
pytest.importorskip("pandas")


def test_prepare_xy_and_one_fold_fusion():
    from tea_gaze.baselines import cross_validate, prepare_xy

    frame = load_frame("zuco_sst_standard")
    texts, gaze, labels = prepare_xy(frame)
    assert len(texts) == 400
    assert gaze.shape == (400, 5)
    assert set(labels.tolist()) == {0, 1, 2}

    result = cross_validate(frame.head(60), name="gaze", n_splits=3, random_state=0)
    assert result.name == "gaze"
    assert len(result.folds) == 3
    assert 0.0 <= result.mean.accuracy <= 1.0
