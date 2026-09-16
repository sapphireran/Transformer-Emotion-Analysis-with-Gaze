"""Human-readable catalogs for features, files, and the two experiment tracks."""

from __future__ import annotations

from dataclasses import dataclass

from . import paths


@dataclass(frozen=True)
class GazeFeature:
    name: str
    aliases: tuple[str, ...]
    unit: str
    meaning: str
    used_in_fusion: bool
    notes: str


# The five columns concatenated into EyeTrackingModel in both training scripts.
FUSION_FEATURES = ("nFixations", "FFD", "GPT", "TRT", "GD")
FUSION_FEATURES_SST = ("nFix", "FFD", "GPT", "TRT", "GD")

GAZE_FEATURES = (
    GazeFeature(
        name="nFixations",
        aliases=("nFix",),
        unit="count / scaled",
        meaning="Number of fixations on the word, or the mean over fixated words in a sentence.",
        used_in_fusion=True,
        notes="Full-SST tables spell this nFix. Raw subject files are counts; training tables are scaled.",
    ),
    GazeFeature(
        name="FFD",
        aliases=(),
        unit="ms / scaled",
        meaning="First Fixation Duration: time of the first fixation on the region.",
        used_in_fusion=True,
        notes="Early lexical processing. Often shorter than GD when the region is refixated.",
    ),
    GazeFeature(
        name="SFD",
        aliases=(),
        unit="ms / scaled",
        meaning="Single Fixation Duration: FFD restricted to regions with exactly one fixation.",
        used_in_fusion=False,
        notes="Present on ZuCo sentence tables, not passed into EyeTrackingModel.",
    ),
    GazeFeature(
        name="GD",
        aliases=(),
        unit="ms / scaled",
        meaning="Gaze Duration (first-pass time): sum of fixations before the eyes leave the region to the right.",
        used_in_fusion=True,
        notes="Also called first-pass dwell time.",
    ),
    GazeFeature(
        name="GPT",
        aliases=(),
        unit="ms / scaled",
        meaning="Go-Past Time: time from first entering a region until it is passed to the right, including regressions.",
        used_in_fusion=True,
        notes="Sensitive to integration difficulty. Often the largest of the duration measures.",
    ),
    GazeFeature(
        name="TRT",
        aliases=(),
        unit="ms / scaled",
        meaning="Total Reading Time: sum of all fixations on the region, including later passes.",
        used_in_fusion=True,
        notes="Late / rereading measure. Correlated with GD and GPT.",
    ),
    GazeFeature(
        name="omissionRate",
        aliases=(),
        unit="proportion / scaled",
        meaning="Fraction of words in the sentence that received no fixation.",
        used_in_fusion=False,
        notes="Sentence-level only. High omission often means skimming.",
    ),
    GazeFeature(
        name="meanPupilSize",
        aliases=(),
        unit="arbitrary / scaled",
        meaning="Mean pupil size over fixations on the region.",
        used_in_fusion=False,
        notes="Raw subject files sit near 800–1000. Never mix raw pupil with z-scored fusion inputs.",
    ),
    GazeFeature(
        name="SentLen",
        aliases=(),
        unit="tokens",
        meaning="Number of words in the sentence as tokenized in the ZuCo MATLAB struct.",
        used_in_fusion=False,
        notes="Useful as an alignment key: subject 3 diverges from the others at id 150.",
    ),
    GazeFeature(
        name="WordLen",
        aliases=(),
        unit="characters",
        meaning="Character length of the cleaned word token.",
        used_in_fusion=False,
        notes="Word-level tables only.",
    ),
)

FEATURE_BY_NAME = {feat.name: feat for feat in GAZE_FEATURES}
for feat in GAZE_FEATURES:
    for alias in feat.aliases:
        FEATURE_BY_NAME[alias] = feat


@dataclass(frozen=True)
class ExperimentTrack:
    key: str
    title: str
    n_sentences: int
    gaze_source: str
    script: str
    combined: str
    train: str
    valid: str
    test: str
    fusion_columns: tuple[str, ...]
    protocol: str
    when_to_use: str


TRACKS = (
    ExperimentTrack(
        key="zuco_sst",
        title="ZuCo–SST overlap",
        n_sentences=400,
        gaze_source="Human eye-tracking, 12 readers, sentence-averaged then z-scored",
        script="model_ZuCo_SST.py",
        combined=str(paths.ZUCO_COMBINED_STANDARD.relative_to(paths.ROOT)),
        train=str(paths.ZUCO_TRAIN.relative_to(paths.ROOT)),
        valid=str(paths.ZUCO_VALID.relative_to(paths.ROOT)),
        test=str(paths.ZUCO_TEST.relative_to(paths.ROOT)),
        fusion_columns=FUSION_FEATURES,
        protocol="Stratified 5-fold on the 400-row combined table (the 320/40/40 files are unused by the script)",
        when_to_use="When the gaze values should be treated as real reading behavior.",
    ),
    ExperimentTrack(
        key="full_sst",
        title="Full SST with projected gaze",
        n_sentences=11853,
        gaze_source="Projected / predicted features, not a 12-reader recording of the full SST",
        script="model_full_SST.py",
        combined=str(paths.SST_COMBINED.relative_to(paths.ROOT)),
        train=str(paths.SST_TRAIN.relative_to(paths.ROOT)),
        valid=str(paths.SST_VALID.relative_to(paths.ROOT)),
        test=str(paths.SST_TEST.relative_to(paths.ROOT)),
        fusion_columns=FUSION_FEATURES_SST,
        protocol="Fixed 80/10/10 split, checkpoint on validation accuracy, then a test loop",
        when_to_use="When you need scale. Do not quote full-SST TRT as a human measurement.",
    ),
)


@dataclass(frozen=True)
class FileNote:
    path: str
    kind: str
    rows: str
    note: str


FILE_NOTES = (
    FileNote("ZuCo_SST_data/combined_sst_et_standard.csv", "train table", "400", "Z-scored gaze + labels. model_ZuCo_SST.py reads this."),
    FileNote("ZuCo_SST_data/combined_sst_et_min_max.csv", "train table", "400", "Same sentences, min-max gaze. Not used by the training scripts."),
    FileNote("ZuCo_SST_data/ssts_ZuCo.csv", "text + label", "400", "No gaze columns. Producer leftover from convert_full_SST.py."),
    FileNote("ZuCo_SST_data/train.csv", "split", "320", "Random 80% split. Valid is not stratified (7/14/19)."),
    FileNote("ZuCo_SST_data/valid.csv", "split", "40", "Too small for a stable accuracy number."),
    FileNote("ZuCo_SST_data/test.csv", "split", "40", "Disjoint from train/valid; unused by 5-fold script."),
    FileNote("ZuCo_et_csv_data/{1-12}_SR.csv", "per reader", "400 (299 for #3)", "Closer to raw ms / pupil. Subject 3 is remapped after id 149."),
    FileNote("ZuCo_et_csv_data/average_data.csv", "mean over readers", "400", "Index-averaged; contaminated for ids 150–398 by subject 3."),
    FileNote("ZuCo_et_csv_data/word/word_averages_v2.csv", "word means", "7129", "400 sentences, Sent_ID like 0_NR. Same index-average issue."),
    FileNote("SST_data/combined_full_sst_et.csv", "train table", "11853", "Z-scored projected gaze. model_full_SST.py uses the split files."),
    FileNote("SST_data/stts_all_sentence_level.csv", "raw SST", "11853", "sentence,POS/NEG/NEU — no header."),
    FileNote("gaze_prediction/data/provo.csv", "external ET", "2659", "PROVO word table with fixProp instead of GD."),
    FileNote("gaze_prediction/data/prediction_test.csv", "predicted ET", "1751", "100 sentences of model-predicted word gaze."),
)
