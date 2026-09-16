from __future__ import annotations

from gaze_emotion_examples.reports import build_dataset_report


def test_dataset_report_contains_core_sections():
    report = build_dataset_report()
    for needle in (
        "# Dataset report",
        "zuco_sst_standard",
        "sst_full",
        "ZuCo ∩ SST, standard-scaled sentence gaze",
        "Split audits",
        "Label mix",
    ):
        assert needle in report
    assert report.count("| key |") == 1
