# Two experiment tracks

This checkout trains the same late-fusion idea in two places. They are not
the same dataset with a different split.

## Track A — ZuCo ∩ SST, recorded gaze

- **File the trainer reads:** `ZuCo_SST_data/combined_sst_et_standard.csv`
- **Size:** 400 movie-review sentences that appear in both ZuCo task 1 (normal
  reading) and Stanford Sentiment Treebank.
- **Labels:** `0` negative (123), `1` neutral (137), `2` positive (140).
- **Gaze:** twelve readers, sentence-level means, then z-scored. The fusion
  head uses `nFixations`, `FFD`, `GPT`, `TRT`, `GD`. Extra columns
  (`omissionRate`, `meanPupilSize`, `SFD`) sit in the CSV and are ignored.
- **Eval:** `model_ZuCo_SST.py` runs `StratifiedKFold(n_splits=5, seed=42)`
  on the 400-row table. The stored `train.csv` / `valid.csv` / `test.csv`
  (80/10/10, seed 42, **not** stratified) are unused by that script.

## Track B — full SST, projected gaze

- **Files the trainer reads:** `SST_data/train_full_sst.csv`,
  `valid_full_sst.csv`, `test_full_sst.csv` (9,482 / 1,185 / 1,186).
- **Parent table:** `SST_data/combined_full_sst_et.csv` (11,853 rows).
- **Labels:** 4,649 / 2,241 / 4,963. Neutral is much rarer than in the ZuCo
  slice.
- **Gaze:** five channels named `nFix` (not `nFixations`), `FFD`, `GPT`,
  `TRT`, `GD`. These are **projected** sentence scores, already z-scored
  (mean ≈ 0, std ≈ 1). They are not twelve-reader means.
- **Eval:** one hold-out loop, `batch_size=256`, five epochs, checkpoint on
  validation accuracy. See `docs/08-trainer-errata.md` before quoting the
  printed test line.

## What is *not* a third track

`gaze_prediction/data/prediction_test_v2.csv` is a word-level 0–100
predictor dump for all 11,853 SST sentences (191,971 tokens). Mean-pooling
those rows does **not** reconstruct `combined_full_sst_et.csv` (different
scale; nFix Pearson r ≈ 0.58 against the sentence table).
`prediction_test.csv` is a 100-sentence slice (ids 300–399). `provo.csv`
is a PROVO-format table (`fixProp` instead of `GD`) and is not wired into
either trainer.

## Why the two tracks exist

Track A asks whether *measured* reading behaviour on the ZuCo overlap
helps a transformer decide sentiment. Track B asks whether a *gaze
predictor* transferred onto the rest of SST does the same thing. Mixing
the two tables, or treating `nFix` and `nFixations` as the same column
without a rename, is a silent bug.
