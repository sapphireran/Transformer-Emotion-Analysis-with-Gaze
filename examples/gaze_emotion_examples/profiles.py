"""Word-level gaze profiles for a single ZuCo or SST sentence."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WordGaze:
    word_id: int
    word: str
    n_fixations: float
    ffd: float
    gpt: float
    trt: float
    gd: float
    word_len: int | None = None

    @property
    def skipped(self) -> bool:
        return self.n_fixations == 0 and self.trt == 0


def sentence_profile(
    words: pd.DataFrame,
    sentence_id,
    sentence_column: str = "Sent_ID",
    word_id_column: str = "Word_ID",
    word_column: str = "Word",
    nfix_column: str = "nFixations",
    ffd_column: str = "FFD",
    gpt_column: str = "GPT",
    trt_column: str = "TRT",
    gd_column: str = "GD",
    length_column: str | None = "WordLen",
) -> list[WordGaze]:
    """Return left-to-right word gaze rows for one sentence id."""
    subset = words.loc[words[sentence_column] == sentence_id].copy()
    if subset.empty:
        # Allow callers to pass a bare integer for ZuCo rows stored as "12_NR".
        subset = words.loc[words[sentence_column] == f"{sentence_id}_NR"].copy()
    if subset.empty:
        raise KeyError(f"No word rows for sentence id {sentence_id!r}")
    subset = subset.sort_values(word_id_column)
    profiles = []
    for mapping in subset.to_dict(orient="records"):
        profiles.append(
            WordGaze(
                word_id=int(mapping[word_id_column]),
                word=str(mapping[word_column]),
                n_fixations=float(mapping[nfix_column]),
                ffd=float(mapping[ffd_column]),
                gpt=float(mapping[gpt_column]),
                trt=float(mapping[trt_column]),
                gd=float(mapping[gd_column]),
                word_len=int(mapping[length_column]) if length_column and length_column in mapping else None,
            )
        )
    return profiles


def reconstruct_sentence(profile: list[WordGaze]) -> str:
    """Join profile tokens with spaces."""
    return " ".join(item.word for item in profile)


def top_words_by_measure(profile: list[WordGaze], measure: str, k: int = 5) -> list[WordGaze]:
    """Highest-valued words for a gaze measure (``trt``, ``gpt``, ``n_fixations``, ...)."""
    allowed = {
        "n_fixations": lambda item: item.n_fixations,
        "ffd": lambda item: item.ffd,
        "gpt": lambda item: item.gpt,
        "trt": lambda item: item.trt,
        "gd": lambda item: item.gd,
    }
    if measure not in allowed:
        raise ValueError(f"measure must be one of {sorted(allowed)}")
    ranked = sorted(profile, key=allowed[measure], reverse=True)
    return ranked[:k]


def profile_frame(profile: list[WordGaze]) -> pd.DataFrame:
    """Convert a profile to a DataFrame for plotting or markdown."""
    return pd.DataFrame(
        [
            {
                "word_id": item.word_id,
                "word": item.word,
                "n_fixations": item.n_fixations,
                "FFD": item.ffd,
                "GPT": item.gpt,
                "TRT": item.trt,
                "GD": item.gd,
                "word_len": item.word_len,
                "skipped": item.skipped,
            }
            for item in profile
        ]
    )
