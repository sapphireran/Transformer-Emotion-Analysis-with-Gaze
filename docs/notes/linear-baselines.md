# Linear baselines on ZuCo ∩ SST

Measured by `examples/06_text_vs_gaze_baselines.py` on
`ZuCo_SST_data/combined_sst_et_standard.csv` (400 sentences, seed 42,
5-fold stratified CV). Classifier is `LogisticRegression` with
`class_weight='balanced'`. Text features are TF-IDF unigrams+bigrams
(`min_df=2`, 4,000 caps). Gaze features are the same five channels the
transformer scripts use: `nFixations, GD, TRT, FFD, GPT`.

This is a linear probe, not RoBERTa. It asks whether sentence-level human
gaze moves a bag-of-n-grams model at all.

## Means

| model | accuracy | macro F1 | weighted F1 |
| --- | ---: | ---: | ---: |
| majority (always positive) | 0.350 | — | — |
| `gaze_only` | 0.348 | 0.336 | 0.337 |
| `text_only` | **0.497** | **0.491** | **0.493** |
| `text_plus_gaze` | 0.472 | 0.465 | 0.469 |

Gaze-only is the majority baseline. Concatenating gaze to TF-IDF **hurts**
by about 2.5 accuracy points. Per-fold table:

| model | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gaze_only acc | 0.388 | 0.388 | 0.312 | 0.300 | 0.350 |
| text_only acc | 0.438 | 0.550 | 0.438 | 0.550 | 0.512 |
| text_plus_gaze acc | 0.450 | 0.538 | 0.450 | 0.462 | 0.462 |

`text_plus_gaze` only beats `text_only` on fold 1 (+1.2). It loses on the
other four folds, including a 8.8-point drop on fold 4.

## How to read this

1. Sentence-level averages of 12 readers do not linearly separate SST-3
   on these 400 reviews. That matches the tiny label-conditional means in
   [gaze-features.md](../gaze-features.md).
2. Late fusion can add noise when the gaze vector is collinear
   (`nFixations`–`TRT` r = 0.96) and weakly related to the label.
3. A transformer might still use gaze as a tie-breaker when the pooler is
   uncertain. These numbers do **not** rule that out. They do say a linear
   head on TF-IDF + z-scored sentence gaze is not enough to claim a gaze
   gain.
4. Word-level structure that the sentence mean destroys — the 1.3 s
   go-past on `decency` in sentence 0 — is a more plausible place to look
   next than another 5-d concat.

Raw fold rows are in `examples/output/text_vs_gaze_baselines.csv` after
you run the script. That file is gitignored as an example artifact; the
table above is the checked-in snapshot.
