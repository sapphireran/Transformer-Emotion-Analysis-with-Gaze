#!/usr/bin/env python3
"""Print row counts, class balance, and length stats for the 400-sentence set."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))

from gazekit.features import class_balance, majority_fraction, summarize_numeric
from gazekit.io import load_sentence_table
from gazekit.paths import default_paths
from gazekit.report import format_frame, section
from gazekit.schema import EXPECTED_ROWS


def build_report(root: Path | None = None) -> dict:
    paths = default_paths(root)
    combined = load_sentence_table(paths.zuco_combined_standard)
    text = load_sentence_table(paths.zuco_text)
    return {
        "n_combined": len(combined),
        "n_text": len(text),
        "expected": EXPECTED_ROWS["zuco_sst_combined"],
        "columns": list(combined.columns),
        "class_balance": class_balance(combined),
        "majority": majority_fraction(combined),
        "length": summarize_numeric(combined, ["n_tokens"]),
        "ids_match_text": set(combined["sentence_id"]) == set(text["sentence_id"]),
        "label_match": bool(
            combined.merge(
                text[["sentence_id", "sentiment_label"]],
                on="sentence_id",
                suffixes=("", "_text"),
            )["sentiment_label"].eq(
                combined.merge(
                    text[["sentence_id", "sentiment_label"]],
                    on="sentence_id",
                    suffixes=("", "_text"),
                )["sentiment_label_text"]
            ).all()
        )
        if "sentiment_label" in text.columns
        else False,
        "token_min": int(combined["n_tokens"].min()),
        "token_max": int(combined["n_tokens"].max()),
        "token_mean": float(combined["n_tokens"].mean()),
    }


def render(report: dict) -> str:
    chunks = [
        section(
            "ZuCo SST combined table",
            f"rows={report['n_combined']} (expected {report['expected']})\n"
            f"text-only rows={report['n_text']}\n"
            f"sentence_id set matches text table: {report['ids_match_text']}\n"
            f"labels match text table: {report['label_match']}\n"
            f"majority-class fraction: {report['majority']:.4f}\n"
            f"token count min/mean/max: "
            f"{report['token_min']} / {report['token_mean']:.2f} / {report['token_max']}",
        ),
        section("Class balance", format_frame(report["class_balance"])),
        section("Length", format_frame(report["length"])),
        section("Columns", ", ".join(report["columns"])),
    ]
    return "\n".join(chunks)


def main() -> None:
    print(render(build_report()))


if __name__ == "__main__":
    main()
