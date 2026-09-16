"""Word-level ↔ sentence-level bridges and label-conditioned gaze profiles."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from . import csvio, labels, numbers, paths
from .schema import SUBJECT_WORD, ZUCO_SENTENCE_GAZE, validate_rows


WORD_DURATION = ("nFixations", "GD", "TRT", "FFD", "SFD", "GPT")


@dataclass(frozen=True)
class SentenceWords:
    sent_id: str
    words: list[str]
    n_zero_nfix: int
    mean_nfix: float
    mean_trt: float


def load_word_averages() -> list[dict[str, str]]:
    rows = csvio.read_dicts(paths.WORD_AVERAGES)
    validate_rows(rows, SUBJECT_WORD)
    return rows


def group_words(rows: Sequence[dict[str, str]] | None = None) -> dict[str, list[dict[str, str]]]:
    rows = list(rows) if rows is not None else load_word_averages()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["Sent_ID"]].append(row)
    for sent_id in grouped:
        grouped[sent_id].sort(key=lambda r: int(float(r["Word_ID"])))
    return dict(grouped)


def sentence_word_profile(sent_id: str, grouped: dict[str, list[dict[str, str]]] | None = None) -> SentenceWords:
    grouped = grouped if grouped is not None else group_words()
    words = grouped[sent_id]
    nfix = [csvio.as_float(row["nFixations"]) for row in words]
    trt = [csvio.as_float(row["TRT"]) for row in words]
    return SentenceWords(
        sent_id=sent_id,
        words=[row["Word"] for row in words],
        n_zero_nfix=sum(1 for value in nfix if value == 0.0),
        mean_nfix=numbers.mean(nfix),
        mean_trt=numbers.mean(trt),
    )


def sent_id_from_index(sentence_id: int, suffix: str = "NR") -> str:
    return f"{sentence_id}_{suffix}"


@dataclass(frozen=True)
class LabelGazeProfile:
    label: int
    name: str
    n: int
    means: dict[str, float]


def label_conditioned_means(
    path=paths.ZUCO_COMBINED_STANDARD,
    features: Sequence[str] = ("nFixations", "FFD", "GPT", "TRT", "GD", "omissionRate"),
) -> list[LabelGazeProfile]:
    rows = csvio.read_dicts(path)
    validate_rows(rows, ZUCO_SENTENCE_GAZE)
    out: list[LabelGazeProfile] = []
    for label in (0, 1, 2):
        subset = [row for row in rows if labels.parse_label(row["sentiment_label"]) == label]
        means = {name: numbers.mean([csvio.as_float(row[name]) for row in subset]) for name in features}
        out.append(LabelGazeProfile(label=label, name=labels.label_name(label), n=len(subset), means=means))
    return out


def feature_label_correlations(
    path=paths.ZUCO_COMBINED_STANDARD,
    features: Sequence[str] = ("nFixations", "FFD", "GPT", "TRT", "GD", "omissionRate"),
) -> dict[str, float]:
    """Pearson between each feature and the integer label 0/1/2.

    This is a coarse monotonic check, not a claim that labels are ordinal.
    """
    rows = csvio.read_dicts(path)
    y = [float(labels.parse_label(row["sentiment_label"])) for row in rows]
    return {name: numbers.pearson([csvio.as_float(row[name]) for row in rows], y) for name in features}


def gaze_collinearity(
    path=paths.ZUCO_COMBINED_STANDARD,
    features: Sequence[str] = ("nFixations", "FFD", "GPT", "TRT", "GD"),
) -> list[tuple[str, str, float]]:
    rows = csvio.read_dicts(path)
    cols = {name: [csvio.as_float(row[name]) for row in rows] for name in features}
    pairs: list[tuple[str, str, float]] = []
    names = list(features)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairs.append((a, b, numbers.pearson(cols[a], cols[b])))
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    return pairs
