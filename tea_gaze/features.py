"""Gaze-feature glossary shared by docs, loaders, and the fusion baselines.

The five columns used by `EyeTrackingModel` in `model_ZuCo_SST.py` and
`model_full_SST.py` are `nFixations`/`nFix`, `FFD`, `GPT`, `TRT`, and `GD`.
The remaining sentence-level columns exist in the ZuCo extracts and are
kept here so reports can describe the full measured set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

SENTIMENT_LABELS: dict[int, str] = {
    0: "NEGATIVE",
    1: "NEUTRAL",
    2: "POSITIVE",
}

LABEL_TO_ID: dict[str, int] = {name: idx for idx, name in SENTIMENT_LABELS.items()}


@dataclass(frozen=True)
class FeatureSpec:
    """One eye-tracking or reading-behavior column."""

    name: str
    aliases: tuple[str, ...]
    description: str
    unit: str
    in_fusion: bool
    typical_raw_range: str

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="nFixations",
        aliases=("nFix", "nfix"),
        description=(
            "Number of fixations on the word or, after sentence aggregation, "
            "the mean fixation count across fixated words. Higher values "
            "usually mean the reader lingered or refixated."
        ),
        unit="count",
        in_fusion=True,
        typical_raw_range="word: 0–6; sentence mean: ~1–4",
    ),
    FeatureSpec(
        name="FFD",
        aliases=(),
        description=(
            "First Fixation Duration: duration of the first fixation on a "
            "word. Sensitive to early lexical access."
        ),
        unit="ms",
        in_fusion=True,
        typical_raw_range="~80–250 ms per word",
    ),
    FeatureSpec(
        name="GPT",
        aliases=(),
        description=(
            "Go-Past Time (regression-path duration): time from first "
            "entering a word until the eyes move to the right of it, "
            "including leftward regressions. Captures integration difficulty."
        ),
        unit="ms",
        in_fusion=True,
        typical_raw_range="often longer than TRT when regressions occur",
    ),
    FeatureSpec(
        name="TRT",
        aliases=(),
        description=(
            "Total Reading Time: sum of all fixation durations on a word, "
            "including later rereading. A late, cumulative difficulty measure."
        ),
        unit="ms",
        in_fusion=True,
        typical_raw_range="word: ~100–400 ms; sentence mean similar",
    ),
    FeatureSpec(
        name="GD",
        aliases=(),
        description=(
            "Gaze Duration (first-pass dwell time): sum of fixations on a "
            "word before the eyes leave it. First-pass processing, excluding "
            "later rereading."
        ),
        unit="ms",
        in_fusion=True,
        typical_raw_range="typically between FFD and TRT",
    ),
    FeatureSpec(
        name="SFD",
        aliases=(),
        description=(
            "Single Fixation Duration: duration when a word received exactly "
            "one fixation. Missing or zero when the word was skipped or "
            "fixated more than once."
        ),
        unit="ms",
        in_fusion=False,
        typical_raw_range="similar to FFD when defined",
    ),
    FeatureSpec(
        name="meanPupilSize",
        aliases=(),
        description=(
            "Mean pupil size while the word or sentence was fixated. Used as "
            "a coarse arousal / cognitive-load proxy. ZuCo stores the "
            "tracker’s native pupil units, not millimeters."
        ),
        unit="tracker units",
        in_fusion=False,
        typical_raw_range="sentence means around 800–900 in this extract",
    ),
    FeatureSpec(
        name="omissionRate",
        aliases=(),
        description=(
            "Fraction of words in the sentence that received no fixation. "
            "High omission is common on short function words."
        ),
        unit="proportion",
        in_fusion=False,
        typical_raw_range="0–0.4 on these movie-review sentences",
    ),
    FeatureSpec(
        name="SentLen",
        aliases=(),
        description="Token count of the sentence in the ZuCo word stream.",
        unit="tokens",
        in_fusion=False,
        typical_raw_range="typically 5–30 tokens",
    ),
    FeatureSpec(
        name="WordLen",
        aliases=(),
        description="Character length of the current word after light cleanup.",
        unit="characters",
        in_fusion=False,
        typical_raw_range="1–15 for most tokens",
    ),
    FeatureSpec(
        name="fixProp",
        aliases=(),
        description=(
            "Fixation probability used by the PROVO reference table "
            "(percent of readers who fixated the word)."
        ),
        unit="percent",
        in_fusion=False,
        typical_raw_range="~50–100 on content words",
    ),
)

_FEATURES_BY_NAME: dict[str, FeatureSpec] = {}
for _spec in FEATURE_SPECS:
    for _name in _spec.all_names:
        _FEATURES_BY_NAME[_name] = _spec
        _FEATURES_BY_NAME[_name.lower()] = _spec

CORE_FUSION_FEATURES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURE_SPECS if spec.in_fusion
)


def list_features(*, fusion_only: bool = False) -> tuple[FeatureSpec, ...]:
    if fusion_only:
        return tuple(spec for spec in FEATURE_SPECS if spec.in_fusion)
    return FEATURE_SPECS


def get_feature(name: str) -> FeatureSpec:
    try:
        return _FEATURES_BY_NAME[name]
    except KeyError as exc:
        known = ", ".join(spec.name for spec in FEATURE_SPECS)
        raise KeyError(f"Unknown gaze feature {name!r}. Known: {known}") from exc


def canonical_name(name: str) -> str:
    return get_feature(name).name


def canonicalize_gaze_columns(columns: Iterable[str]) -> dict[str, str]:
    """Map actual CSV headers onto canonical feature names.

    Example: `{'nFix': 'nFixations'}` when a full-SST file uses the short name.
    """
    mapping: dict[str, str] = {}
    for column in columns:
        key = column.strip()
        spec = _FEATURES_BY_NAME.get(key) or _FEATURES_BY_NAME.get(key.lower())
        if spec is not None:
            mapping[column] = spec.name
    return mapping


def resolve_fusion_columns(columns: Iterable[str]) -> list[str]:
    """Return the CSV column names that fill the five fusion features.

    Raises `KeyError` if any fusion feature is missing.
    """
    available = canonicalize_gaze_columns(columns)
    inverse: dict[str, str] = {}
    for original, canonical in available.items():
        inverse.setdefault(canonical, original)

    missing = [name for name in CORE_FUSION_FEATURES if name not in inverse]
    if missing:
        raise KeyError(
            "CSV is missing fusion features: "
            + ", ".join(missing)
            + f". Have: {sorted(columns)}"
        )
    return [inverse[name] for name in CORE_FUSION_FEATURES]


def fusion_feature_frame(frame: "object") -> "object":
    """Return a DataFrame with canonical fusion columns in model order.

    `frame` is a pandas DataFrame. The import is local so feature docs can
    load this module without pandas installed.
    """
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("fusion_feature_frame expects a pandas DataFrame")
    source_columns = resolve_fusion_columns(frame.columns)
    out = frame.loc[:, source_columns].apply(pd.to_numeric, errors="coerce")
    out.columns = list(CORE_FUSION_FEATURES)
    return out


def label_name(label: int | str) -> str:
    return SENTIMENT_LABELS[int(label)]


def describe_features(specs: Iterable[FeatureSpec] | None = None) -> str:
    rows = []
    for spec in specs or FEATURE_SPECS:
        used = "yes" if spec.in_fusion else "no"
        alias = ", ".join(spec.aliases) if spec.aliases else "—"
        rows.append(
            f"| `{spec.name}` | {alias} | {spec.unit} | {used} | {spec.description} |"
        )
    header = (
        "| Feature | Aliases | Unit | In fusion model | Description |\n"
        "|---|---|---|---|---|\n"
    )
    return header + "\n".join(rows)


def feature_glossary() -> Mapping[str, FeatureSpec]:
    return {spec.name: spec for spec in FEATURE_SPECS}
