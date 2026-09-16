#!/usr/bin/env python3
"""Forward-pass demo of the late-fusion wiring, without downloading RoBERTa."""

from __future__ import annotations

from _util import maybe_write, parser
from zuco_lab import csvio, paths
from zuco_lab.catalogs import FUSION_FEATURES
from zuco_lab.fusion import LateFusionToy, architecture_lines, tokenize
from zuco_lab.reports import join_sections, md_heading, md_table


def _row_gaze(row: dict[str, str]) -> list[float]:
    return [float(row[name]) for name in FUSION_FEATURES]


def build(n: int = 6) -> str:
    model = LateFusionToy.seeded(7)
    rows = csvio.read_dicts(paths.ZUCO_COMBINED_STANDARD)[:n]
    pred_rows = []
    for row in rows:
        tokens = tokenize(row["sentence"])
        pred, probs, _ = model.predict(tokens, _row_gaze(row))
        pred_rows.append(
            (
                row["sentence_id"],
                row["sentiment_label"],
                pred,
                probs[0],
                probs[1],
                probs[2],
                row["sentence"][:64] + ("…" if len(row["sentence"]) > 64 else ""),
            )
        )
    return join_sections(
        [
            md_heading("CPU late-fusion wiring demo", 1),
            "```\n" + "\n".join(architecture_lines()) + "\n```",
            (
                "Weights are seeded and tiny. Predictions are not an accuracy "
                "claim. This only shows that a 5-d gaze vector and a hashed "
                "sentence can share a concatenated classifier, the same way "
                "``EyeTrackingModel`` concatenates ``pooler_output`` with "
                "``Linear(5 → 16)``."
            ),
            md_heading(f"First {n} ZuCo rows, seed=7"),
            md_table(
                ("id", "gold", "pred", "p_neg", "p_neu", "p_pos", "text"),
                pred_rows,
            ),
        ]
    )


def main() -> None:
    p = parser("Run the stdlib late-fusion toy on real ZuCo rows.")
    p.add_argument("--n", type=int, default=6)
    args = p.parse_args()
    text = build(args.n)
    print(text)
    maybe_write(args, "08_cpu_fusion.md", text)


if __name__ == "__main__":
    main()
