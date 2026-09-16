"""Resolved paths for the personal datasets already in this repo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (parent of the `tea_gaze` package)."""
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DatasetRef:
    """A named CSV that examples and docs can load without hard-coding paths."""

    key: str
    relative_path: str
    kind: str
    description: str
    gaze_source: str

    @property
    def path(self) -> Path:
        return repo_root() / self.relative_path

    def exists(self) -> bool:
        return self.path.is_file()


DATASETS: dict[str, DatasetRef] = {
    "zuco_sst_standard": DatasetRef(
        key="zuco_sst_standard",
        relative_path="ZuCo_SST_data/combined_sst_et_standard.csv",
        kind="sentence",
        description=(
            "400 ZuCo Task-1 sentences joined to Stanford Sentiment Treebank "
            "labels. Eye-tracking columns are z-scored across sentences."
        ),
        gaze_source="measured",
    ),
    "zuco_sst_minmax": DatasetRef(
        key="zuco_sst_minmax",
        relative_path="ZuCo_SST_data/combined_sst_et_min_max.csv",
        kind="sentence",
        description=(
            "Same 400 ZuCo+SST rows as `zuco_sst_standard`, but gaze columns "
            "are min-max scaled to [0, 1] instead of standardized."
        ),
        gaze_source="measured",
    ),
    "zuco_sst_labels": DatasetRef(
        key="zuco_sst_labels",
        relative_path="ZuCo_SST_data/ssts_ZuCo.csv",
        kind="sentence",
        description="ZuCo sentence text and 3-class sentiment labels only.",
        gaze_source="none",
    ),
    "zuco_sst_train": DatasetRef(
        key="zuco_sst_train",
        relative_path="ZuCo_SST_data/train.csv",
        kind="sentence",
        description="80/10/10 hold-out split of the standardized ZuCo+SST table.",
        gaze_source="measured",
    ),
    "zuco_sst_valid": DatasetRef(
        key="zuco_sst_valid",
        relative_path="ZuCo_SST_data/valid.csv",
        kind="sentence",
        description="Validation slice of the standardized ZuCo+SST table.",
        gaze_source="measured",
    ),
    "zuco_sst_test": DatasetRef(
        key="zuco_sst_test",
        relative_path="ZuCo_SST_data/test.csv",
        kind="sentence",
        description="Test slice of the standardized ZuCo+SST table.",
        gaze_source="measured",
    ),
    "zuco_et_average": DatasetRef(
        key="zuco_et_average",
        relative_path="ZuCo_et_csv_data/average_data.csv",
        kind="sentence",
        description=(
            "Sentence-level eye-tracking averages across the 12 ZuCo readers, "
            "in raw units (milliseconds / counts)."
        ),
        gaze_source="measured",
    ),
    "zuco_et_standard": DatasetRef(
        key="zuco_et_standard",
        relative_path="ZuCo_et_csv_data/standard_scaled_average_data.csv",
        kind="sentence",
        description="Sentence-level ZuCo averages after StandardScaler.",
        gaze_source="measured",
    ),
    "zuco_et_minmax": DatasetRef(
        key="zuco_et_minmax",
        relative_path="ZuCo_et_csv_data/min_max_scaled_average_data.csv",
        kind="sentence",
        description="Sentence-level ZuCo averages after MinMaxScaler.",
        gaze_source="measured",
    ),
    "zuco_word_averages": DatasetRef(
        key="zuco_word_averages",
        relative_path="ZuCo_et_csv_data/word/word_averages_v2.csv",
        kind="word",
        description=(
            "Word-level ZuCo Task-1 averages (12 readers) for the same 400 "
            "normal-reading sentences."
        ),
        gaze_source="measured",
    ),
    "full_sst_combined": DatasetRef(
        key="full_sst_combined",
        relative_path="SST_data/combined_full_sst_et.csv",
        kind="sentence",
        description=(
            "Full Stanford Sentiment Treebank sentences with predicted "
            "sentence-level gaze features (`nFix`, `GD`, `TRT`, `FFD`, `GPT`)."
        ),
        gaze_source="predicted",
    ),
    "full_sst_train": DatasetRef(
        key="full_sst_train",
        relative_path="SST_data/train_full_sst.csv",
        kind="sentence",
        description="Training split of the full SST + predicted-gaze table.",
        gaze_source="predicted",
    ),
    "full_sst_valid": DatasetRef(
        key="full_sst_valid",
        relative_path="SST_data/valid_full_sst.csv",
        kind="sentence",
        description="Validation split of the full SST + predicted-gaze table.",
        gaze_source="predicted",
    ),
    "full_sst_test": DatasetRef(
        key="full_sst_test",
        relative_path="SST_data/test_full_sst.csv",
        kind="sentence",
        description="Test split of the full SST + predicted-gaze table.",
        gaze_source="predicted",
    ),
    "gaze_prediction_test": DatasetRef(
        key="gaze_prediction_test",
        relative_path="gaze_prediction/data/prediction_test.csv",
        kind="word",
        description="Word-level predicted gaze for a 100-sentence SST sample.",
        gaze_source="predicted",
    ),
    "provo": DatasetRef(
        key="provo",
        relative_path="gaze_prediction/data/provo.csv",
        kind="word",
        description=(
            "PROVO word-level reading measures used as an external gaze "
            "reference while building the predictor."
        ),
        gaze_source="measured",
    ),
}


def subject_et_paths() -> list[Path]:
    """Return the 12 per-reader sentence-level ZuCo CSVs, ordered 1..12."""
    folder = repo_root() / "ZuCo_et_csv_data"
    return [folder / f"{index}_SR.csv" for index in range(1, 13)]


def require_dataset(key: str) -> DatasetRef:
    if key not in DATASETS:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset {key!r}. Known keys: {known}")
    ref = DATASETS[key]
    if not ref.exists():
        raise FileNotFoundError(f"Dataset {key!r} is missing at {ref.path}")
    return ref
