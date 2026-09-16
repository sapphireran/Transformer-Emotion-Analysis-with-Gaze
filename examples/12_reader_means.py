#!/usr/bin/env python3
"""Per-reader sentence-level means (pupil, omission, TRT) — subject 12's tiny pupil."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sidecar.load import load_all_subject_sentence  # noqa: E402
from sidecar.reports import markdown_table, write_text  # noqa: E402


def main() -> int:
    subs = load_all_subject_sentence()
    rows = []
    for i, df in subs.items():
        rows.append(
            {
                "subject": i,
                "n_sentences": len(df),
                "id_min": int(df["id"].min()),
                "id_max": int(df["id"].max()),
                "mean_SentLen": float(df["SentLen"].mean()),
                "mean_omission": float(df["omissionRate"].mean()),
                "mean_nFixations": float(df["nFixations"].mean()),
                "mean_pupil": float(df["meanPupilSize"].mean()),
                "mean_TRT": float(df["TRT"].mean()),
                "mean_FFD": float(df["FFD"].mean()),
                "mean_GPT": float(df["GPT"].mean()),
            }
        )
    table = pd.DataFrame(rows)
    pupil = table["mean_pupil"]
    text = "\n".join(
        [
            "# Reader-level sentence means",
            "",
            "Task-1 sentence tables (`ZuCo_et_csv_data/{1..12}_SR.csv`). Subject 3 is",
            "the 299-row packed file. Subject 12's mean pupil is an outlier on the low",
            "side (~298 vs ~800–1300). That survives into `average_data.csv` because",
            "the average is a plain mean across whatever rows share a DataFrame index.",
            "",
            markdown_table(table),
            "",
            f"Pupil min/median/max across readers: {pupil.min():.1f} / "
            f"{pupil.median():.1f} / {pupil.max():.1f}.",
            "",
            "Omission rate is the fraction of words in the sentence with no reported",
            "fixation. Subject 3 is again the high-omission reader, which matches the",
            "word-level skip table.",
            "",
        ]
    )
    out = write_text("reader_means.md", text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
