#!/usr/bin/env python3
"""How much do the 12 ZuCo readers disagree on the same sentences?

The fusion head never sees a subject id — `get_average_sentence_level.py`
collapses the 12 raw tables first. This script looks at that collapse:

  - per-subject means of the five fusion features (raw milliseconds)
  - coefficient of variation across subjects
  - pairwise TRT correlation on the 299 sentences *all* subjects have
    (subject 3 is truncated; we align on id 0..298)

If readers barely correlate, a subject-averaged 5-d vector is a blurry
"typical reader." If they correlate strongly, averaging is mostly denoising.

    python3 examples/07_subject_variance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import as_float_column, read_csv, table, write_text

FEATURES = ["nFixations", "FFD", "GPT", "TRT", "GD"]
N_SUBJECTS = 12
ALIGN_ROWS = 299  # subject 3 only has ids 0..298


def load_subject(i: int) -> tuple[np.ndarray, np.ndarray]:
    header, rows = read_csv(f"ZuCo_et_csv_data/{i}_SR.csv")
    ids = as_float_column(rows, header, "id").astype(int)
    feats = np.column_stack([as_float_column(rows, header, name) for name in FEATURES])
    return ids, feats


def main() -> int:
    subjects = []
    for i in range(1, N_SUBJECTS + 1):
        ids, feats = load_subject(i)
        subjects.append((i, ids, feats))

    mean_rows = []
    for i, ids, feats in subjects:
        mean_rows.append((f"subject {i:02d}", str(len(ids)), *[f"{feats[:, j].mean():.1f}" for j in range(len(FEATURES))]))

    # Align on the 299 ids that exist for everyone.
    aligned = []
    for i, ids, feats in subjects:
        order = {int(s): k for k, s in enumerate(ids)}
        mat = np.vstack([feats[order[s]] for s in range(ALIGN_ROWS)])
        aligned.append(mat)
    stacked = np.stack(aligned, axis=0)  # (12, 299, 5)

    cv_rows = []
    for j, name in enumerate(FEATURES):
        # per-sentence std / |mean| across subjects, then average those CVs
        sent_mean = stacked[:, :, j].mean(axis=0)
        sent_std = stacked[:, :, j].std(axis=0, ddof=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(np.abs(sent_mean) < 1e-8, np.nan, sent_std / np.abs(sent_mean))
        cv_rows.append(
            (
                name,
                f"{float(np.nanmean(cv)):.3f}",
                f"{float(np.nanmedian(cv)):.3f}",
                f"{float(np.nanpercentile(cv, 90)):.3f}",
            )
        )

    # Pairwise subject correlations on TRT (index 3)
    trt = stacked[:, :, FEATURES.index("TRT")]
    corr = np.corrcoef(trt)
    # mean off-diagonal
    off = corr[~np.eye(len(corr), dtype=bool)]
    pair_rows = []
    for a in range(N_SUBJECTS):
        pair_rows.append((f"s{a + 1:02d}", *[f"{corr[a, b]:.2f}" for b in range(N_SUBJECTS)]))

    # Compare subject-mean vs committed average_data.csv on the aligned ids
    avg_header, avg_rows = read_csv("ZuCo_et_csv_data/average_data.csv")
    avg_ids = as_float_column(avg_rows, avg_header, "id").astype(int)
    avg_trt = as_float_column(avg_rows, avg_header, "TRT")
    avg_map = {int(i): v for i, v in zip(avg_ids, avg_trt)}
    recomputed = stacked[:, :, FEATURES.index("TRT")].mean(axis=0)
    committed = np.array([avg_map[i] for i in range(ALIGN_ROWS)])
    mae = float(np.mean(np.abs(recomputed - committed)))
    # subject 3 missing tail: committed average of ids 299..399 is 11-reader mean
    tail_ids = [i for i in avg_ids if i >= ALIGN_ROWS]
    tail_note = (
        f"average_data.csv ids >= {ALIGN_ROWS}: {len(tail_ids)} rows "
        "(subject 3 absent; 11-reader mean)"
    )

    lines = [
        "# Per-subject gaze variance (raw ZuCo sentence tables)",
        "",
        "Units are the raw exports: counts for nFixations, milliseconds for the rest.",
        f"Aligned correlation / CV use ids 0..{ALIGN_ROWS - 1} so subject 3 can vote.",
        "",
        "## mean feature by subject (all of that subject's rows)",
        table(["subject", "n", *FEATURES], mean_rows),
        "",
        "## across-subject coefficient of variation, then averaged over sentences",
        "CV = std_subjects / |mean_subjects| for each sentence, then mean/median/p90.",
        table(["feature", "mean CV", "median CV", "p90 CV"], cv_rows),
        "",
        "## pairwise subject correlation on TRT (aligned 299 sentences)",
        f"mean off-diagonal r = {float(off.mean()):.3f}  "
        f"min = {float(off.min()):.3f}  max = {float(off.max()):.3f}",
        table(["", *[f"s{i:02d}" for i in range(1, N_SUBJECTS + 1)]], pair_rows),
        "",
        "## does average_data.csv match a simple 12-reader mean?",
        f"MAE on aligned TRT (recomputed mean vs committed): {mae:.4f} ms",
        tail_note,
        "",
        "## how to read this",
        "- Large CV and modest pairwise r → averaging is aggressive; a lot of",
        "  reader-specific behavior is thrown away before training.",
        "- High pairwise r → readers linger on the same sentences; the 5-d",
        "  vector is a stable 'this sentence is hard' signal.",
        "- A non-zero MAE against average_data.csv is expected if the averaging",
        "  script converted 0 → NaN before the mean (this script does not).",
        "",
    ]
    text = "\n".join(lines)
    print(text)
    write_text("07_subject_variance.txt", text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
