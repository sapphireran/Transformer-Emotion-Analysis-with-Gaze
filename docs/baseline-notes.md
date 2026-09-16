# Baseline notes from this checkout

Numbers below come from the CPU scripts on the committed CSVs. They are
not transformer results. Re-run `python3 examples/run_all.py` after any
table change.

## Inventory

- ZuCo combined: **400** sentences (123 / 137 / 140 negative / neutral / positive).
- ZuCo splits: 320 / 40 / 40 with all three classes present.
- Full SST: **11853** sentences (4649 / 2241 / 4963), splits 9482 / 1185 / 1186.
- Word averages: 7129 rows, 400 sentences, 55 `unknown` tokens.
- Predicted word-level v2: 191971 rows, **complete** `sentence_id` overlap with full SST.
- Subject 3: 299 rows (`id` 0–298). Other eleven subjects have 400 rows.

`examples/join_zuco_sst.py` rebuilds the standard and min-max joins with
fusion-column max `|Δ|` at floating-point noise (`4.4e-16` on one TRT column).

## Real ZuCo gaze vs. label

`f_classif` on the five fusion columns is weak (largest `GD` F=2.21,
p=0.11). `SFD`, which is **not** in the classifier, is the strongest extra
column (F=4.03, p=0.019). Pearson `|r|` among `nFixations`, `GPT`, and
`TRT` is above 0.91.

Cross-subject TRT median CV is 0.28. Subject 3’s leave-one-out correlation
against the other-subject mean is 0.33; subject 5 is 0.71. The
twelve-subject average is therefore a blurry sentence vector, especially
on late measures.

## Projected full-SST gaze vs. label

`f_classif` looks significant (F≈25–32) **and** the five columns are almost
the same number (`|r|` often 0.98–0.999). That is one latent direction, not
five reading-time measures. Positive sentences sit slightly below zero on
every projected feature; negative and neutral sit slightly above.

## Linear classification

| Track | model | acc | weighted F1 | macro F1 |
| --- | --- | --- | --- | --- |
| ZuCo | gaze | 0.350 | 0.331 | 0.310 |
| ZuCo | length | 0.275 | 0.239 | 0.250 |
| ZuCo | tfidf | 0.525 | 0.520 | 0.524 |
| ZuCo | tfidf+gaze | 0.500 | 0.500 | 0.500 |
| Full SST | gaze | 0.345 | 0.340 | 0.321 |
| Full SST | length | 0.331 | 0.284 | 0.277 |
| Full SST | tfidf | 0.621 | 0.628 | 0.548 |
| Full SST | tfidf+gaze | 0.619 | 0.626 | 0.546 |

`fusion_toy.py` (16-d random projection of gaze, concat onto TF-IDF):

| Track | text F1 | concat F1 | Δ |
| --- | --- | --- | --- |
| ZuCo | 0.520 | 0.500 | −0.020 |
| Full SST | 0.624 | 0.628 | +0.004 |

Personal takeaway: **gaze-only is near chance; adding the five sentence
features to a linear text model does not help.** Any later RoBERTa fusion
gain has to be non-linear or a variance artifact on the 40-row ZuCo test
slice.

## Word-level sanity

Long alphabetic tokens attract more fixations than a small function-word
list on ZuCo averages (1.58 vs 0.55), predicted v2 (24.3 vs 16.2), and
PROVO (22.8 vs 6.1). Word ids are contiguous on ZuCo and predicted v2.
PROVO `nFix` / `GPT` / `TRT` can be slightly negative; that table is a
reference corpus, not a ZuCo export.
