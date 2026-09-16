from gaze_emotion.constants import (
    FUSION_GAZE_FEATURES,
    FUSION_GAZE_FEATURES_FULL_SST,
    LABEL_NAME_TO_ID,
    fusion_feature_names,
    label_name,
)
import pytest


def test_label_roundtrip():
    for name, idx in LABEL_NAME_TO_ID.items():
        assert label_name(idx) == name
        assert label_name(name) == name
        assert label_name(str(idx)) == name


def test_unknown_label_raises():
    with pytest.raises(KeyError):
        label_name(9)
    with pytest.raises(KeyError):
        label_name("HAPPY")


def test_fusion_feature_styles():
    assert fusion_feature_names("zuco") == FUSION_GAZE_FEATURES
    assert fusion_feature_names("full_sst") == FUSION_GAZE_FEATURES_FULL_SST
    assert "nFixations" in FUSION_GAZE_FEATURES
    assert "nFix" in FUSION_GAZE_FEATURES_FULL_SST
    assert set(FUSION_GAZE_FEATURES) - {"nFixations"} == set(FUSION_GAZE_FEATURES_FULL_SST) - {"nFix"}


def test_fusion_style_rejects_unknown():
    with pytest.raises(ValueError):
        fusion_feature_names("eeg")
