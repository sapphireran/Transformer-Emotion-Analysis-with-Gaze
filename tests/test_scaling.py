from __future__ import annotations

import pandas as pd

from gaze_emotion_examples.io import load_dataset
from gaze_emotion_examples.scaling import columnwise_rank_agreement, scaler_report


def test_standard_table_looks_standard():
    frame = load_dataset("zuco_sst_standard")
    report = scaler_report(
        frame,
        ["omissionRate", "nFixations", "meanPupilSize", "GD", "TRT", "FFD", "SFD", "GPT"],
    )
    assert report.standard_like
    assert not report.minmax_like


def test_minmax_table_looks_minmax():
    frame = load_dataset("zuco_sst_minmax")
    report = scaler_report(
        frame,
        ["omissionRate", "nFixations", "meanPupilSize", "GD", "TRT", "FFD", "SFD", "GPT"],
    )
    assert report.minmax_like
    assert not report.standard_like


def test_raw_table_is_neither():
    frame = load_dataset("zuco_sentence_raw")
    report = scaler_report(frame, ["nFixations", "GD", "TRT", "FFD", "GPT"])
    assert not report.standard_like
    assert not report.minmax_like
    assert report.to_frame().loc[0, "mean"] > 1.0


def test_monotone_scaling_keeps_ranks():
    raw = pd.DataFrame({"nFixations": [1.0, 2.0, 4.0, 7.0]})
    scaled = pd.DataFrame({"nFixations": (raw["nFixations"] - raw["nFixations"].mean()) / raw["nFixations"].std()})
    agreement = columnwise_rank_agreement(raw, scaled, ["nFixations"])
    assert agreement["nFixations"] == 1.0
