"""Inventory of the personal CSVs checked into this repository."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .paths import data_path


@dataclass(frozen=True)
class DatasetSpec:
    """One on-disk table plus the columns examples should treat as gaze."""

    key: str
    relative_path: str
    title: str
    kind: str
    description: str
    label_column: str | None = None
    id_column: str | None = None
    text_column: str | None = None
    gaze_columns: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def path(self):
        return data_path(self.relative_path)


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="zuco_sst_standard",
        relative_path="ZuCo_SST_data/combined_sst_et_standard.csv",
        title="ZuCo ∩ SST, standard-scaled sentence gaze",
        kind="sentence",
        description=(
            "400 movie-review sentences that appear in ZuCo Task 1 (normal reading) "
            "joined to SST-3 labels. Gaze columns are z-scored after averaging "
            "across subjects."
        ),
        label_column="sentiment_label",
        id_column="sentence_id",
        text_column="sentence",
        gaze_columns=(
            "omissionRate",
            "nFixations",
            "meanPupilSize",
            "GD",
            "TRT",
            "FFD",
            "SFD",
            "GPT",
        ),
        notes=(
            "This is the table model_ZuCo_SST.py trains on.",
            "Labels: 0=negative, 1=neutral, 2=positive.",
        ),
    ),
    DatasetSpec(
        key="zuco_sst_minmax",
        relative_path="ZuCo_SST_data/combined_sst_et_min_max.csv",
        title="ZuCo ∩ SST, min-max-scaled sentence gaze",
        kind="sentence",
        description=(
            "Same 400 sentences and labels as zuco_sst_standard, with each gaze "
            "column independently rescaled to [0, 1]."
        ),
        label_column="sentiment_label",
        id_column="sentence_id",
        text_column="sentence",
        gaze_columns=(
            "omissionRate",
            "nFixations",
            "meanPupilSize",
            "GD",
            "TRT",
            "FFD",
            "SFD",
            "GPT",
        ),
    ),
    DatasetSpec(
        key="zuco_sst_text",
        relative_path="ZuCo_SST_data/ssts_ZuCo.csv",
        title="ZuCo ∩ SST text and labels only",
        kind="sentence",
        description="Sentence text and SST-3 labels before gaze columns are joined.",
        label_column="sentiment_label",
        id_column="sentence_id",
        text_column="sentence",
    ),
    DatasetSpec(
        key="zuco_sst_train",
        relative_path="ZuCo_SST_data/train.csv",
        title="ZuCo ∩ SST 80% train split",
        kind="split",
        description="Random 80/10/10 split of the standard-scaled ZuCo table (seed 42).",
        label_column="sentiment_label",
        id_column="sentence_id",
        text_column="sentence",
        gaze_columns=(
            "omissionRate",
            "nFixations",
            "meanPupilSize",
            "GD",
            "TRT",
            "FFD",
            "SFD",
            "GPT",
        ),
        notes=("The ZuCo training script uses 5-fold CV on the full 400 rows, not this split.",),
    ),
    DatasetSpec(
        key="zuco_sst_valid",
        relative_path="ZuCo_SST_data/valid.csv",
        title="ZuCo ∩ SST 10% valid split",
        kind="split",
        description="Held-out validation slice of the standard-scaled ZuCo table.",
        label_column="sentiment_label",
        id_column="sentence_id",
        text_column="sentence",
        gaze_columns=(
            "omissionRate",
            "nFixations",
            "meanPupilSize",
            "GD",
            "TRT",
            "FFD",
            "SFD",
            "GPT",
        ),
    ),
    DatasetSpec(
        key="zuco_sst_test",
        relative_path="ZuCo_SST_data/test.csv",
        title="ZuCo ∩ SST 10% test split",
        kind="split",
        description="Held-out test slice of the standard-scaled ZuCo table.",
        label_column="sentiment_label",
        id_column="sentence_id",
        text_column="sentence",
        gaze_columns=(
            "omissionRate",
            "nFixations",
            "meanPupilSize",
            "GD",
            "TRT",
            "FFD",
            "SFD",
            "GPT",
        ),
    ),
    DatasetSpec(
        key="zuco_sentence_raw",
        relative_path="ZuCo_et_csv_data/average_data.csv",
        title="Subject-averaged ZuCo sentence gaze (raw units)",
        kind="sentence",
        description=(
            "Per-sentence means across the 12 ZuCo readers, still in milliseconds "
            "and fixation counts. No SST labels."
        ),
        id_column="id",
        gaze_columns=(
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
        notes=("Subject 3 contributes only the first 299 rows; later rows average 11 readers.",),
    ),
    DatasetSpec(
        key="zuco_word_raw",
        relative_path="ZuCo_et_csv_data/word/word_averages_v2.csv",
        title="Subject-averaged ZuCo word gaze (raw units)",
        kind="word",
        description="7,129 word tokens from the same 400 sentences, averaged over readers.",
        id_column="id",
        text_column="Word",
        gaze_columns=(
            "nFixations",
            "meanPupilSize",
            "GD",
            "TRT",
            "FFD",
            "SFD",
            "GPT",
            "WordLen",
        ),
        notes=("Sent_ID uses the ZuCo NR suffix, e.g. 0_NR.",),
    ),
    DatasetSpec(
        key="sst_full",
        relative_path="SST_data/combined_full_sst_et.csv",
        title="Full SST-3 with sentence-level predicted gaze",
        kind="sentence",
        description=(
            "11,853 SST sentences with five z-scored gaze channels. Those channels "
            "come from aggregating word-level predictions, not from ZuCo readers."
        ),
        label_column="sentiment_label",
        id_column="sentence_id",
        text_column="sentence",
        gaze_columns=("nFix", "GD", "TRT", "FFD", "GPT"),
        notes=("This is the table model_full_SST.py trains on.",),
    ),
    DatasetSpec(
        key="sst_train",
        relative_path="SST_data/train_full_sst.csv",
        title="Full SST 80% train split",
        kind="split",
        description="Random 80/10/10 split of sst_full (seed 42).",
        label_column="sentiment_label",
        id_column="sentence_id",
        text_column="sentence",
        gaze_columns=("nFix", "GD", "TRT", "FFD", "GPT"),
    ),
    DatasetSpec(
        key="sst_valid",
        relative_path="SST_data/valid_full_sst.csv",
        title="Full SST 10% valid split",
        kind="split",
        description="Held-out validation slice of sst_full.",
        label_column="sentiment_label",
        id_column="sentence_id",
        text_column="sentence",
        gaze_columns=("nFix", "GD", "TRT", "FFD", "GPT"),
    ),
    DatasetSpec(
        key="sst_test",
        relative_path="SST_data/test_full_sst.csv",
        title="Full SST 10% test split",
        kind="split",
        description="Held-out test slice of sst_full.",
        label_column="sentiment_label",
        id_column="sentence_id",
        text_column="sentence",
        gaze_columns=("nFix", "GD", "TRT", "FFD", "GPT"),
    ),
    DatasetSpec(
        key="sst_word_predicted",
        relative_path="gaze_prediction/data/prediction_test_v2.csv",
        title="Word-level predicted gaze for every SST sentence",
        kind="word",
        description=(
            "191,971 word rows covering all 11,853 SST sentences. Values are "
            "model-predicted reading measures, not human recordings."
        ),
        id_column="sentence_id",
        text_column="word",
        gaze_columns=("nFix", "FFD", "GPT", "TRT", "GD"),
    ),
    DatasetSpec(
        key="provo_word",
        relative_path="gaze_prediction/data/provo.csv",
        title="Provo corpus word-level gaze (gaze-prediction source)",
        kind="word",
        description=(
            "2,659 words / 134 sentences from the Provo corpus, used as a "
            "natural-reading source for the gaze predictor."
        ),
        id_column="sentence_id",
        text_column="word",
        gaze_columns=("nFix", "FFD", "GPT", "TRT", "fixProp"),
    ),
)


_BY_KEY = {spec.key: spec for spec in DATASETS}


def list_datasets(kind: str | None = None) -> tuple[DatasetSpec, ...]:
    """Return all specs, or only those whose kind matches."""
    if kind is None:
        return DATASETS
    return tuple(spec for spec in DATASETS if spec.kind == kind)


def get_dataset(key: str) -> DatasetSpec:
    """Look up a spec by catalog key."""
    try:
        return _BY_KEY[key]
    except KeyError as exc:
        known = ", ".join(sorted(_BY_KEY))
        raise KeyError(f"Unknown dataset {key!r}. Known keys: {known}") from exc


def require_keys(keys: Iterable[str]) -> tuple[DatasetSpec, ...]:
    """Resolve several keys, failing if any file is missing on disk."""
    specs = tuple(get_dataset(key) for key in keys)
    missing = [spec.relative_path for spec in specs if not spec.path.exists()]
    if missing:
        raise FileNotFoundError("Missing data files: " + ", ".join(missing))
    return specs
