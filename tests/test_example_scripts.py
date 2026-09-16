"""Smoke-run the example CLIs against the checked-in CSVs."""

from __future__ import annotations

from examples.compare_scalings import main as compare_main
from examples.inspect_datasets import main as inspect_main
from examples.sentence_gaze_walkthrough import main as walk_main
from examples.split_balance import main as split_main


def test_inspect_strict():
    assert inspect_main(["--strict"]) == 0


def test_compare_scalings():
    assert compare_main() == 0


def test_split_balance():
    assert split_main() == 0


def test_walkthrough_sentence_three(capsys):
    assert walk_main(["--sentence-id", "3", "--subjects", "2"]) == 0
    out = capsys.readouterr().out
    assert "unintentionally" in out
    assert "measured" in out
    assert "+4.1272" in out or "+4.13" in out
