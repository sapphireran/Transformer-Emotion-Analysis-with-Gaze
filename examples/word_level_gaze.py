#!/usr/bin/env python3
"""Inspect word-level ZuCo averages and predicted full-SST gaze.

The classifier is sentence-level, but the projection pipeline is word-level.
This script checks that those tables are internally consistent:

- each sentence has contiguous word_id values starting at 0
- measure ranges are finite
- short function words have lower nFix than long content words (a weak
  sanity check, not a linguistic claim)
- predicted v2 rows cover the same sentence_id span as full SST when possible

Usage (from repo root):

    python3 examples/word_level_gaze.py
    python3 examples/word_level_gaze.py --write examples/sample_outputs/word_level_gaze.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from paths import PRED_WORD_V2, PROVO, SST_COMBINED, ZUCO_WORD_AVERAGES

FUNCTION_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "as",
    "by",
    "from",
}

CONTENT_RE = re.compile(r"^[A-Za-z]{5,}$")


def _contiguous_word_ids(df: pd.DataFrame, sent_col: str, word_col: str) -> dict:
    broken = 0
    checked = 0
    for _, group in df.groupby(sent_col, sort=False):
        ids = group[word_col].to_numpy()
        checked += 1
        expected = np.arange(len(ids))
        # Some exports store Word_ID already 0..n-1 but not sorted on disk.
        if not np.array_equal(np.sort(ids), expected):
            broken += 1
    return {"sentences_checked": checked, "noncontiguous_or_gapped": broken}


def _range_report(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        out[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "non_finite": int((~np.isfinite(series.to_numpy(dtype=float))).sum()),
        }
    return out


def _length_effect(df: pd.DataFrame, word_col: str, nfix_col: str) -> dict:
    words = df[word_col].astype(str).str.lower()
    nfix = pd.to_numeric(df[nfix_col], errors="coerce")
    function_mask = words.isin(FUNCTION_WORDS)
    content_mask = words.str.match(CONTENT_RE.pattern) & ~function_mask
    return {
        "function_n": int(function_mask.sum()),
        "function_mean_nfix": float(nfix[function_mask].mean()) if function_mask.any() else None,
        "long_content_n": int(content_mask.sum()),
        "long_content_mean_nfix": float(nfix[content_mask].mean()) if content_mask.any() else None,
    }


def analyze_zuco_words() -> dict:
    df = pd.read_csv(ZUCO_WORD_AVERAGES)
    sent_ids = df["Sent_ID"].astype(str)
    payload = {
        "path": str(ZUCO_WORD_AVERAGES),
        "rows": int(len(df)),
        "unique_sentences": int(sent_ids.nunique()),
        "unknown_words": int((df["Word"].astype(str) == "unknown").sum()),
        "word_id_integrity": _contiguous_word_ids(df, "Sent_ID", "Word_ID"),
        "ranges": _range_report(
            df, ["nFixations", "FFD", "GPT", "TRT", "GD", "SFD", "meanPupilSize", "WordLen"]
        ),
        "length_effect": _length_effect(df, "Word", "nFixations"),
        "words_per_sentence": {
            "mean": float(df.groupby("Sent_ID").size().mean()),
            "max": int(df.groupby("Sent_ID").size().max()),
        },
    }
    return payload


def analyze_predictions() -> dict:
    df = pd.read_csv(PRED_WORD_V2)
    sst = pd.read_csv(SST_COMBINED)
    pred_ids = set(df["sentence_id"].astype(int))
    sst_ids = set(sst["sentence_id"].astype(int))
    payload = {
        "path": str(PRED_WORD_V2),
        "rows": int(len(df)),
        "unique_sentences": int(df["sentence_id"].nunique()),
        "sst_sentences": int(len(sst_ids)),
        "pred_ids_missing_from_sst": int(len(pred_ids - sst_ids)),
        "sst_ids_missing_from_pred": int(len(sst_ids - pred_ids)),
        "word_id_integrity": _contiguous_word_ids(df, "sentence_id", "word_id"),
        "ranges": _range_report(df, ["nFix", "FFD", "GPT", "TRT", "GD"]),
        "length_effect": _length_effect(df, "word", "nFix"),
        "words_per_sentence": {
            "mean": float(df.groupby("sentence_id").size().mean()),
            "max": int(df.groupby("sentence_id").size().max()),
        },
    }
    return payload


def analyze_provo() -> dict:
    df = pd.read_csv(PROVO)
    return {
        "path": str(PROVO),
        "rows": int(len(df)),
        "unique_sentences": int(df["sentence_id"].nunique()),
        "columns": df.columns.tolist(),
        "ranges": _range_report(df, [c for c in ["nFix", "FFD", "GPT", "TRT", "fixProp"] if c in df.columns]),
        "length_effect": _length_effect(df, "word", "nFix"),
    }


def _print(title: str, payload: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"rows={payload.get('rows')} sentences={payload.get('unique_sentences')}")
    if "word_id_integrity" in payload:
        integrity = payload["word_id_integrity"]
        print(
            "word_id gaps: "
            f"{integrity['noncontiguous_or_gapped']} / {integrity['sentences_checked']}"
        )
    if "length_effect" in payload:
        effect = payload["length_effect"]
        print(
            "mean nFix function vs long-content: "
            f"{effect['function_mean_nfix']:.3f} (n={effect['function_n']}) vs "
            f"{effect['long_content_mean_nfix']:.3f} (n={effect['long_content_n']})"
        )
    if "sst_ids_missing_from_pred" in payload:
        print(
            "id overlap vs full SST: "
            f"pred-not-in-sst={payload['pred_ids_missing_from_sst']} "
            f"sst-not-in-pred={payload['sst_ids_missing_from_pred']}"
        )
    print("ranges:")
    for col, stats in payload["ranges"].items():
        print(
            f"  {col:16s} min={stats['min']:.3f} max={stats['max']:.3f} "
            f"mean={stats['mean']:.3f} non_finite={stats['non_finite']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None)
    args = parser.parse_args()

    reports = {
        "zuco_word_averages": analyze_zuco_words(),
        "predicted_word_v2": analyze_predictions(),
        "provo": analyze_provo(),
    }
    _print("ZuCo word averages v2", reports["zuco_word_averages"])
    _print("predicted word-level gaze v2", reports["predicted_word_v2"])
    _print("PROVO reference", reports["provo"])

    print("\nSanity: long content words should attract more fixations than 'the'/'of'.")
    print("If predicted v2 misses SST ids, sentence-level projection used another join.")

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nwrote {args.write}")


if __name__ == "__main__":
    main()
