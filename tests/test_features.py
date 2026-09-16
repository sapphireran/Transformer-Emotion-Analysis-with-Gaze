from tea_gaze.features import (
    CORE_FUSION_FEATURES,
    canonicalize_gaze_columns,
    get_feature,
    resolve_fusion_columns,
)
from tea_gaze.text import tokenize


def test_fusion_feature_order_matches_training_scripts():
    assert CORE_FUSION_FEATURES == ("nFixations", "FFD", "GPT", "TRT", "GD")


def test_nfix_alias_maps_to_nfixations():
    mapping = canonicalize_gaze_columns(["nFix", "FFD", "GPT", "TRT", "GD"])
    assert mapping["nFix"] == "nFixations"
    assert resolve_fusion_columns(mapping) == ["nFix", "FFD", "GPT", "TRT", "GD"]


def test_feature_glossary_covers_fusion_set():
    for name in CORE_FUSION_FEATURES:
        spec = get_feature(name)
        assert spec.in_fusion
        assert spec.unit


def test_tokenizer_keeps_simple_words():
    tokens = tokenize("Slow, silly and unintentionally hilarious.")
    assert tokens == ["slow", "silly", "and", "unintentionally", "hilarious"]
