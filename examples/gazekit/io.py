"""CSV loaders that normalize gaze column names to the canonical five."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import CANONICAL_GAZE, SENTIMENT_TO_INT, ZUCO_TO_CANONICAL


def _read_csv(path: Path | str, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"expected CSV at {path}")
    return pd.read_csv(path, **kwargs)


def canonicalize_gaze(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Rename ``nFix`` → ``nFixations`` when present. Leave other columns alone."""
    out = df.copy() if copy else df
    rename = {src: dst for src, dst in ZUCO_TO_CANONICAL.items() if src in out.columns and src != dst}
    if rename:
        # Avoid a collision if both names somehow exist.
        for src, dst in list(rename.items()):
            if dst in out.columns:
                out[dst] = out[src]
                out = out.drop(columns=[src])
                del rename[src]
        if rename:
            out = out.rename(columns=rename)
    return out


def add_token_count(df: pd.DataFrame, text_col: str = "sentence") -> pd.DataFrame:
    """Whitespace token count as a portable length control."""
    if text_col not in df.columns:
        raise KeyError(f"{text_col} is not in columns {list(df.columns)}")
    out = df.copy()
    out["n_tokens"] = out[text_col].fillna("").astype(str).str.split().str.len()
    return out


def load_sentence_table(path: Path | str, add_length: bool = True) -> pd.DataFrame:
    """Load any sentence-level fused table and canonicalize gaze names."""
    df = canonicalize_gaze(_read_csv(path))
    if add_length and "sentence" in df.columns:
        df = add_token_count(df)
    return df


def load_headerless_sst(path: Path | str) -> pd.DataFrame:
    """Load ``stts_all_sentence_level.csv`` (no header, string labels)."""
    df = _read_csv(path, header=None, names=["sentence", "sentiment_name"])
    df["sentiment_name"] = df["sentiment_name"].str.strip()
    df["sentiment_label"] = df["sentiment_name"].map(SENTIMENT_TO_INT)
    if df["sentiment_label"].isna().any():
        unknown = sorted(df.loc[df["sentiment_label"].isna(), "sentiment_name"].unique())
        raise ValueError(f"unknown sentiment strings: {unknown}")
    df["sentiment_label"] = df["sentiment_label"].astype(int)
    return add_token_count(df)


def load_word_table(path: Path | str) -> pd.DataFrame:
    df = canonicalize_gaze(_read_csv(path))
    if "Sent_ID" in df.columns and "sentence_id" not in df.columns:
        df["sentence_id"] = (
            df["Sent_ID"].astype(str).str.split("_").str[0].astype(int)
        )
    if "Word" in df.columns and "word" not in df.columns:
        df["word"] = df["Word"]
    return df


def require_gaze(df: pd.DataFrame, columns=CANONICAL_GAZE) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"missing gaze columns {missing}; have {list(df.columns)}")
    return df


def gaze_matrix(df: pd.DataFrame, columns=CANONICAL_GAZE):
    """Return an ``(N, 5)`` float64 ndarray in canonical order."""
    require_gaze(df, columns)
    return df.loc[:, list(columns)].to_numpy(dtype="float64")
