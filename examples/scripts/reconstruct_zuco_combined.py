"""Show that ZuCo combined CSVs are text ⨝ scaled sentence gaze."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import sys

from teag_examples.paths import examples_output
from teag_examples.pipeline import max_abs_diff, reconstruct_zuco_combined
from teag_examples.reports import write_json, write_markdown
from teag_examples.schema import ZUCO_JOIN_GAZE_COLS


def main(argv: list[str] | None = None) -> int:
    del argv
    rows = []
    payload = {}
    for scaling in ("standard", "min_max"):
        reconstructed, committed = reconstruct_zuco_combined(scaling)
        err = max_abs_diff(reconstructed, committed, ZUCO_JOIN_GAZE_COLS)
        labels_ok = (
            reconstructed.sort_values("sentence_id")["sentiment_label"]
            .reset_index(drop=True)
            .equals(
                committed.sort_values("sentence_id")["sentiment_label"].reset_index(drop=True)
            )
        )
        text_ok = (
            reconstructed.sort_values("sentence_id")["sentence"]
            .reset_index(drop=True)
            .equals(committed.sort_values("sentence_id")["sentence"].reset_index(drop=True))
        )
        payload[scaling] = {
            "n_reconstructed": int(len(reconstructed)),
            "n_committed": int(len(committed)),
            "max_abs_gaze_diff": err,
            "labels_match": bool(labels_ok),
            "sentences_match": bool(text_ok),
        }
        rows.append(
            f"- **{scaling}**: n={len(reconstructed)}, max |Δ gaze|={err:.3e}, "
            f"labels_match={labels_ok}, sentences_match={text_ok}"
        )
    md = (
        "# Reconstruct ZuCo combined tables\n\n"
        "`ssts_ZuCo.csv` inner-joined to scaled `average_data` on "
        "`sentence_id == id`, dropping `SentLen`.\n\n"
        + "\n".join(rows)
        + "\n"
    )
    write_markdown(md, examples_output() / "reconstruct_zuco.md")
    write_json(payload, examples_output() / "reconstruct_zuco.json")
    sys.stdout.write(md + "\n")
    if any(v["max_abs_gaze_diff"] > 1e-12 or not v["labels_match"] for v in payload.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
