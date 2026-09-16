#!/usr/bin/env python3
"""Build a standalone HTML lab book from the committed CSVs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gazebook.baselines import cv_majority, cv_ridge, hash_bow
from gazebook.csvio import float_col, read_dicts, table_to_array
from gazebook.paths import repo_root
from gazebook.remap import (
    NFIX_CONTAMINATED,
    SENTLEN_CONTAMINATED,
    WORD_ALIGN_ROWS,
    compact_to_original,
    contamination_report,
    load_subject_tables,
    published_column,
)
from gazebook.reports import html_page, html_table, write_text
from gazebook.schema import (
    FULL_SST_LAST_BATCH,
    FULL_SST_TEST_ROWS,
    SST_GAZE,
    SST_LABEL_COUNTS,
    ZUCO_GAZE,
    ZUCO_LABEL_COUNTS,
    inventory,
)
from gazebook.stats import condition_number_corr, icc1, pairwise_reader_r
from gazebook.tokens import KNOWN_GLUED, glued_hits, load_word_averages


def main() -> int:
    root = repo_root()
    _, zuco = read_dicts(root / "ZuCo_SST_data/combined_sst_et_standard.csv")
    _, sst = read_dicts(root / "SST_data/combined_full_sst_et.csv")
    tables = load_subject_tables(root)
    words = load_word_averages(root)

    pub = published_column(root, "nFixations")
    slen = published_column(root, "SentLen")
    nfix = contamination_report(tables, pub, "nFixations")
    slen_c = contamination_report(tables, slen, "SentLen")

    X_clean = np.array(
        [[float(tables[r][i]["nFixations"]) for r in range(12)] for i in range(150)],
        dtype=np.float64,
    )
    mean_r, min_r, max_r, _ = pairwise_reader_r(X_clean)
    icc = icc1(X_clean)

    zX = table_to_array(zuco, ZUCO_GAZE)
    sX = table_to_array(sst, SST_GAZE)
    z_cond = condition_number_corr(zX)
    s_cond = condition_number_corr(sX)

    y = float_col(zuco, "sentiment_label").astype(int)
    maj = cv_majority(y)
    gaze = cv_ridge(zX, y)
    text = cv_ridge(hash_bow([r["sentence"] for r in zuco]), y)
    fused = cv_ridge(np.concatenate([hash_bow([r["sentence"] for r in zuco]), zX], axis=1), y)

    inv_rows = [
        [item["path"], item["expected_rows"], item.get("rows", "—"), item.get("notes", "")]
        for item in inventory(root)
    ]
    glued = glued_hits(words)

    body = []
    body.append('<p class="lede">Personal forensic notes for the ZuCo ∩ SST and full-SST gaze-fusion tracks. Numbers are read from this checkout.</p>')

    body.append("<section><h2>Two tracks</h2>")
    body.append(
        html_table(
            ["track", "n", "gaze", "labels 0/1/2", "eval"],
            [
                [
                    "ZuCo ∩ SST",
                    400,
                    "recorded, 12 readers, then averaged",
                    f"{ZUCO_LABEL_COUNTS[0]}/{ZUCO_LABEL_COUNTS[1]}/{ZUCO_LABEL_COUNTS[2]}",
                    "Stratified 5-fold in model_ZuCo_SST.py",
                ],
                [
                    "Full SST",
                    11853,
                    "projected (nFix, not nFixations)",
                    f"{SST_LABEL_COUNTS[0]}/{SST_LABEL_COUNTS[1]}/{SST_LABEL_COUNTS[2]}",
                    "80/10/10 hold-out in model_full_SST.py",
                ],
            ],
        )
    )
    body.append("</section>")

    body.append("<section><h2>Reader 3 compaction</h2>")
    body.append(
        f"<p>MATLAB task-1 subject 2 drops sentences 150–249 and 399, then writes "
        f"<code>3_SR.csv</code> with compact ids 0–298. Compact 150 is original "
        f"<strong>{compact_to_original(150)}</strong>. The published sentence mean "
        f"is a positional 0→NaN average, so nFixations moves on "
        f"<strong>{nfix.n_changed}</strong> sentences (target {NFIX_CONTAMINATED}) "
        f"and SentLen — which is not even a gaze feature — moves on "
        f"<strong>{slen_c.n_changed}</strong> (target {SENTLEN_CONTAMINATED}). "
        f"Worst nFixations shift is id {nfix.worst_id}, |Δ|={nfix.max_abs:.3f}.</p>"
    )
    body.append(
        f"<p>Word streams stay aligned for the first <strong>{WORD_ALIGN_ROWS}</strong> "
        f"tokens (sentences 0–149), then <code>word_averages_v2.csv</code> averages "
        f"different words at the same row index.</p>"
    )
    body.append(
        f"<p>On the clean overlap, nFixations mean pairwise r = {mean_r:.3f} "
        f"(range {min_r:.3f}–{max_r:.3f}), ICC(1) = {icc:.3f}.</p>"
    )
    body.append("</section>")

    body.append("<section><h2>Projected-gaze rank</h2>")
    body.append(
        f"<p>Correlation-matrix condition number: recorded ZuCo <strong>{z_cond:.0f}</strong> "
        f"vs projected full SST <strong>{s_cond:.0f}</strong>. The five full-SST "
        f"channels are almost one number repeated (except GD). A Linear(5→16) "
        f"gaze stem cannot invent four extra degrees of freedom that are not in the table.</p>"
    )
    body.append("</section>")

    body.append("<section><h2>CPU teaching floor (ZuCo, 5-fold ridge)</h2>")
    body.append(
        html_table(
            ["model", "mean acc", "mean weighted F1"],
            [
                ["majority", maj.mean_acc, maj.mean_f1],
                ["gaze only", gaze.mean_acc, gaze.mean_f1],
                ["hashed text", text.mean_acc, text.mean_f1],
                ["text + gaze", fused.mean_acc, fused.mean_f1],
            ],
        )
    )
    body.append("<p>Not a RoBERTa score. Linear gaze is a weak standalone signal on these 400 rows.</p>")
    body.append("</section>")

    body.append("<section><h2>Word-token artifacts</h2>")
    body.append(
        html_table(
            ["Sent_ID", "token in CSV", "intended reading"],
            [[sid, tok, orig] for sid, tok, orig in KNOWN_GLUED],
        )
    )
    body.append(f"<p>Recovered {len(glued)} / {len(KNOWN_GLUED)} of those tokens in word_averages_v2.csv.</p>")
    body.append("</section>")

    body.append("<section><h2>Trainer errata (left in place)</h2>")
    body.append("<ul>")
    body.append(
        f"<li><code>model_full_SST.py</code> overwrites <code>all_preds</code> each test batch. "
        f"Printed test metrics use the last {FULL_SST_LAST_BATCH} of {FULL_SST_TEST_ROWS} rows.</li>"
    )
    body.append("<li><code>model_ZuCo_SST.py</code> ignores the stored 80/10/10 CSVs and runs 5-fold CV.</li>")
    body.append("<li>The stored ZuCo valid split is 7 / 14 / 19 (neg / neu / pos) on 40 rows.</li>")
    body.append("<li>Export scripts still point at <code>et_csv_data/</code> and a Windows MATLAB path.</li>")
    body.append("</ul></section>")

    body.append("<section><h2>Catalogued tables</h2>")
    body.append(html_table(["path", "expected", "seen", "notes"], inv_rows))
    body.append("</section>")

    html = html_page("Gaze-sentiment lab book", "\n".join(body))
    dest = root / "docs/labbook.html"
    write_text(dest, html)
    write_text(root / "examples/output/10_labbook_pointer.md", f"Wrote {dest}\n")
    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
