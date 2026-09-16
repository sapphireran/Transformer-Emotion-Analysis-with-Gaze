"""Checked-in CSV locations and the inventory the inspectors walk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]

GazeKind = Literal["measured", "projected", "auxiliary", "none"]
Level = Literal["sentence", "word", "mixed"]


@dataclass(frozen=True)
class DatasetSpec:
    """One file the personal docs treat as canonical."""

    key: str
    relative: str
    gaze_kind: GazeKind
    level: Level
    expected_rows: int
    expected_cols: tuple[str, ...] | None
    has_header: bool = True
    notes: str = ""

    @property
    def path(self) -> Path:
        return REPO_ROOT / self.relative

    def exists(self) -> bool:
        return self.path.is_file()


def _specs() -> tuple[DatasetSpec, ...]:
    return (
        DatasetSpec(
            key="zuco_text",
            relative="ZuCo_SST_data/ssts_ZuCo.csv",
            gaze_kind="none",
            level="sentence",
            expected_rows=400,
            expected_cols=("sentence_id", "sentence", "sentiment_label"),
            notes="400 ZuCo Task-1 sentences with ternary SST labels.",
        ),
        DatasetSpec(
            key="zuco_standard",
            relative="ZuCo_SST_data/combined_sst_et_standard.csv",
            gaze_kind="measured",
            level="sentence",
            expected_rows=400,
            expected_cols=(
                "sentence_id",
                "sentence",
                "sentiment_label",
                "omissionRate",
                "nFixations",
                "meanPupilSize",
                "GD",
                "TRT",
                "FFD",
                "SFD",
                "GPT",
            ),
            notes="Experiment A table (z-scored measured gaze).",
        ),
        DatasetSpec(
            key="zuco_minmax",
            relative="ZuCo_SST_data/combined_sst_et_min_max.csv",
            gaze_kind="measured",
            level="sentence",
            expected_rows=400,
            expected_cols=(
                "sentence_id",
                "sentence",
                "sentiment_label",
                "omissionRate",
                "nFixations",
                "meanPupilSize",
                "GD",
                "TRT",
                "FFD",
                "SFD",
                "GPT",
            ),
            notes="Same 400 rows as zuco_standard, min-max gaze.",
        ),
        DatasetSpec(
            key="zuco_train",
            relative="ZuCo_SST_data/train.csv",
            gaze_kind="measured",
            level="sentence",
            expected_rows=320,
            expected_cols=None,
            notes="80% hold-out; the CV trainer does not use this file.",
        ),
        DatasetSpec(
            key="zuco_valid",
            relative="ZuCo_SST_data/valid.csv",
            gaze_kind="measured",
            level="sentence",
            expected_rows=40,
            expected_cols=None,
        ),
        DatasetSpec(
            key="zuco_test",
            relative="ZuCo_SST_data/test.csv",
            gaze_kind="measured",
            level="sentence",
            expected_rows=40,
            expected_cols=None,
        ),
        DatasetSpec(
            key="zuco_sentence_average",
            relative="ZuCo_et_csv_data/average_data.csv",
            gaze_kind="measured",
            level="sentence",
            expected_rows=400,
            expected_cols=(
                "id",
                "SentLen",
                "omissionRate",
                "nFixations",
                "meanPupilSize",
                "GD",
                "TRT",
                "FFD",
                "SFD",
                "GPT",
            ),
            notes="Subject-mean gaze in raw tracker units.",
        ),
        DatasetSpec(
            key="zuco_sentence_standard",
            relative="ZuCo_et_csv_data/standard_scaled_average_data.csv",
            gaze_kind="measured",
            level="sentence",
            expected_rows=400,
            expected_cols=None,
        ),
        DatasetSpec(
            key="zuco_sentence_minmax",
            relative="ZuCo_et_csv_data/min_max_scaled_average_data.csv",
            gaze_kind="measured",
            level="sentence",
            expected_rows=400,
            expected_cols=None,
        ),
        DatasetSpec(
            key="zuco_word_average",
            relative="ZuCo_et_csv_data/word/word_averages_v2.csv",
            gaze_kind="measured",
            level="word",
            expected_rows=7129,
            expected_cols=(
                "id",
                "Sent_ID",
                "Word_ID",
                "Word",
                "nFixations",
                "meanPupilSize",
                "GD",
                "TRT",
                "FFD",
                "SFD",
                "GPT",
                "WordLen",
            ),
        ),
        DatasetSpec(
            key="sst_combined",
            relative="SST_data/combined_full_sst_et.csv",
            gaze_kind="projected",
            level="sentence",
            expected_rows=11853,
            expected_cols=(
                "sentence_id",
                "sentence",
                "sentiment_label",
                "nFix",
                "GD",
                "TRT",
                "FFD",
                "GPT",
            ),
            notes="Experiment B table. Gaze is projected, not measured.",
        ),
        DatasetSpec(
            key="sst_train",
            relative="SST_data/train_full_sst.csv",
            gaze_kind="projected",
            level="sentence",
            expected_rows=9482,
            expected_cols=None,
        ),
        DatasetSpec(
            key="sst_valid",
            relative="SST_data/valid_full_sst.csv",
            gaze_kind="projected",
            level="sentence",
            expected_rows=1185,
            expected_cols=None,
        ),
        DatasetSpec(
            key="sst_test",
            relative="SST_data/test_full_sst.csv",
            gaze_kind="projected",
            level="sentence",
            expected_rows=1186,
            expected_cols=None,
        ),
        DatasetSpec(
            key="sst_raw",
            relative="SST_data/stts_all_sentence_level.csv",
            gaze_kind="none",
            level="sentence",
            expected_rows=11853,
            expected_cols=None,
            has_header=False,
            notes="Headerless SST dump with string polarities. Same 11853 rows as combined_full_sst_et.csv.",
        ),
        DatasetSpec(
            key="sst_word_skeleton",
            relative="SST_data/sst_et_test.csv",
            gaze_kind="none",
            level="word",
            expected_rows=191971,
            expected_cols=(
                "sentence_id",
                "word_id",
                "word",
                "nFix",
                "FFD",
                "GPT",
                "TRT",
                "GD",
            ),
            notes="Word rows with zeroed gaze, used as a prediction request.",
        ),
        DatasetSpec(
            key="pred_word_v2",
            relative="gaze_prediction/data/prediction_test_v2.csv",
            gaze_kind="projected",
            level="word",
            expected_rows=191971,
            expected_cols=(
                "sentence_id",
                "word_id",
                "word",
                "nFix",
                "FFD",
                "GPT",
                "TRT",
                "GD",
            ),
        ),
        DatasetSpec(
            key="pred_word_small",
            relative="gaze_prediction/data/prediction_test.csv",
            gaze_kind="projected",
            level="word",
            expected_rows=1751,
            expected_cols=None,
        ),
        DatasetSpec(
            key="provo",
            relative="gaze_prediction/data/provo.csv",
            gaze_kind="auxiliary",
            level="word",
            expected_rows=2659,
            expected_cols=(
                "sentence_id",
                "word_id",
                "word",
                "nFix",
                "FFD",
                "GPT",
                "TRT",
                "fixProp",
            ),
            notes="PROVO extract. fixProp is not gaze duration.",
        ),
    )


_SPEC_TUPLE = _specs()
_SPEC_BY_KEY = {spec.key: spec for spec in _SPEC_TUPLE}


def iter_dataset_specs() -> Iterator[DatasetSpec]:
    return iter(_SPEC_TUPLE)


def spec_by_key(key: str) -> DatasetSpec:
    try:
        return _SPEC_BY_KEY[key]
    except KeyError as exc:
        known = ", ".join(sorted(_SPEC_BY_KEY))
        raise KeyError(f"unknown dataset key {key!r}; known: {known}") from exc


def subject_sentence_csv(subject: int) -> Path:
    """1-based subject file with 400 sentence-level rows."""
    if subject < 1 or subject > 12:
        raise ValueError(f"subject must be 1-12, got {subject}")
    return REPO_ROOT / "ZuCo_et_csv_data" / f"{subject}_SR.csv"


def subject_word_csv(subject: int) -> Path:
    if subject < 1 or subject > 12:
        raise ValueError(f"subject must be 1-12, got {subject}")
    return REPO_ROOT / "ZuCo_et_csv_data" / "word" / f"{subject}_SR.csv"


def expected_subject_sentence_rows(subject: int) -> int:
    """Task 1 subject index 2 (file 3_SR.csv) dropped 101 MATLAB sentences."""
    return 299 if subject == 3 else 400


def expected_subject_word_rows(subject: int) -> int:
    return 5293 if subject == 3 else 7129
