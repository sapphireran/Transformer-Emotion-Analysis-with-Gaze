"""Smoke-import every example script's ``build_report`` against the real CSVs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "examples" / "scripts"


def _load(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "filename",
    [
        "01_explore_zuco_sst.py",
        "02_gaze_feature_report.py",
        "03_gaze_only_baseline.py",
        "04_split_sanity_check.py",
        "05_word_level_scanpath.py",
        "06_fusion_shape_check.py",
    ],
)
def test_script_build_report(filename):
    module = _load(filename)
    report = module.build_report()
    assert isinstance(report, dict)
    rendered = module.render(report)
    assert len(rendered) > 20


def test_explore_report_flags_expected_size():
    report = _load("01_explore_zuco_sst.py").build_report()
    assert report["n_combined"] == 400
    assert report["ids_match_text"] is True
    assert report["label_match"] is True


def test_fusion_script_logits_finite():
    report = _load("06_fusion_shape_check.py").build_report(batch_size=4)
    assert report["logits_finite"] is True
    assert report["constants"]["fused_width"] == 784
