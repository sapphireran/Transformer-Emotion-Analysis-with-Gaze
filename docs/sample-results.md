# Sample results from the CPU examples

Recorded on this branch after `PYTHONPATH=. python3 examples/run_all.py`. These are **not** transformer scores.

## Dataset inventory

All 16 named CSVs in `tea_gaze.paths.DATASETS` were present.

| Key | Rows | Gaze |
|---|---:|---|
| `zuco_sst_standard` | 400 | measured |
| `zuco_sst_train` / `valid` / `test` | 320 / 40 / 40 | measured |
| `zuco_word_averages` | 7129 | measured |
| `full_sst_train` / `valid` / `test` | 9482 / 1185 / 1186 | predicted |
| `gaze_prediction_test` | 1751 | predicted |
| `provo` | 2659 | measured |

ZuCo+SST labels: 123 negative, 137 neutral, 140 positive.

## Raw sentence-mean gaze (`average_data.csv`)

| Feature | mean | stdev | min | max |
|---|---:|---:|---:|---:|
| nFixations | 1.687 | 0.310 | 1.205 | 3.583 |
| FFD (ms) | 116.9 | 8.0 | 101.5 | 165.9 |
| GD (ms) | 141.4 | 21.5 | 107.6 | 272.7 |
| TRT (ms) | 202.6 | 47.9 | 131.1 | 427.0 |
| GPT (ms) | 241.8 | 56.7 | 153.2 | 586.9 |

nFixations vs TRT Pearson r = 0.96. Mean pairwise reader agreement on nFixations is **r = 0.274** (strongest 4&12 = 0.453; weakest 3&10 = 0.023 on 299 shared sentences).

## Text vs gaze vs fusion (5-fold, seed 42)

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| text (TF-IDF + LR) | 0.4675 | 0.4648 | 0.4675 | 0.4621 |
| gaze (5 features) | 0.3475 | 0.3426 | 0.3475 | 0.3369 |
| fusion | 0.4725 | 0.4722 | 0.4725 | 0.4679 |

Gaze-only is near chance on a 3-class task. Fusion adds about half a point of F1 over text. That is a real personal result: on these 400 z-scored sentences the five reading-time means are highly collinear and weakly tied to the label, so a linear head has little extra work to do.

Gaze-only coefficients (all 400 rows) put a positive `GPT` weight on NEGATIVE (`+0.408`) and a positive `nFixations` weight on NEUTRAL (`+0.494`). Treat those as descriptive, not causal.

## Sentence walkthrough

Short examples used by `04_sentence_walkthrough.py`:

| id | Gold | Predicted | Note |
|---|---|---|---|
| 43 | NEGATIVE | NEGATIVE | “How did it ever get made?” |
| 214 | NEUTRAL | NEUTRAL | “It never is, not fully.” — very high z-scored nFix/GPT/TRT |
| 339 | POSITIVE | POSITIVE | “An exhilarating experience.” |
