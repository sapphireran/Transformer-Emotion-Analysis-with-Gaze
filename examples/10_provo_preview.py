#!/usr/bin/env python3
"""Side-by-side look at PROVO vs predicted word-level gaze tables."""

from __future__ import annotations

from collections import Counter

from _util import maybe_write, parser
from zuco_lab import csvio, numbers, paths
from zuco_lab.reports import join_sections, md_heading, md_table
from zuco_lab.schema import PRED_WORD, PROVO_WORD, validate_rows


def _summary(rows, numeric, id_col="sentence_id"):
    n_sent = len({row[id_col] for row in rows})
    stats = []
    for name in numeric:
        vals = [float(row[name]) for row in rows]
        stats.append((name, numbers.mean(vals), numbers.pstdev(vals), min(vals), max(vals)))
    return n_sent, stats


def build() -> str:
    provo = csvio.read_dicts(paths.PROVO)
    pred = csvio.read_dicts(paths.PRED_TEST)
    validate_rows(provo, PROVO_WORD)
    validate_rows(pred, PRED_WORD)

    p_sents, p_stats = _summary(provo, ("nFix", "FFD", "GPT", "TRT", "fixProp"))
    d_sents, d_stats = _summary(pred, ("nFix", "FFD", "GPT", "TRT", "GD"))

    p_words = Counter(int(row["word_id"]) for row in provo)
    d_words = Counter(int(row["word_id"]) for row in pred)

    return join_sections(
        [
            md_heading("PROVO vs predicted word gaze", 1),
            (
                "These tables live under ``gaze_prediction/data/`` and are not "
                "read by the two training scripts. They are the leftover of a "
                "word-level prediction experiment (PROVO as a public ET corpus, "
                "plus a predicted file whose sentence ids start at 300)."
            ),
            md_heading("Coverage"),
            md_table(
                ("table", "rows", "sentences", "first sentence_id", "last sentence_id", "extra column"),
                [
                    (
                        "provo.csv",
                        len(provo),
                        p_sents,
                        min(int(r["sentence_id"]) for r in provo),
                        max(int(r["sentence_id"]) for r in provo),
                        "fixProp",
                    ),
                    (
                        "prediction_test.csv",
                        len(pred),
                        d_sents,
                        min(int(r["sentence_id"]) for r in pred),
                        max(int(r["sentence_id"]) for r in pred),
                        "GD",
                    ),
                ],
            ),
            md_heading("PROVO feature ranges"),
            md_table(("feature", "mean", "std", "min", "max"), p_stats),
            md_heading("Predicted-test feature ranges"),
            md_table(("feature", "mean", "std", "min", "max"), d_stats),
            md_heading("Do not mix these scales with ZuCo z-scores"),
            (
                f"PROVO nFix sits near {p_stats[0][1]:.1f} with a percent-like "
                f"fixProp (mean {p_stats[4][1]:.1f}). Predicted nFix is also a "
                "large unstandardized count. The ZuCo training table is z-scored "
                "around 0. Feeding PROVO columns into ``EyeTrackingModel`` without "
                "a scaler would let nFix dominate the Linear(5 → 16) layer."
            ),
            (
                f"Word-id histograms are just a sanity check that neither file is "
                f"a single sentence: PROVO unique word_ids={len(p_words)}, "
                f"predicted unique word_ids={len(d_words)}."
            ),
        ]
    )


def main() -> None:
    args = parser("Summarize PROVO and predicted word-level gaze tables.").parse_args()
    text = build()
    print(text)
    maybe_write(args, "10_provo_preview.md", text)


if __name__ == "__main__":
    main()
