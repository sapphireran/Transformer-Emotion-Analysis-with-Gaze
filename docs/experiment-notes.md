# Experiment notes (computed from the checked-in CSVs)

These numbers come from `pandas` on the files in git, plus the cheap baselines
in `examples/`. They are personal lab notes, not a paper table.

## Class balance

ZuCo 400 is almost balanced (123 / 137 / 140). Full SST is not: positive and
negative dominate, neutral is 18.9% of 11,853 sentences. Any “accuracy”
number on full SST has to beat a majority-positive baseline of
4963 / 11853 ≈ **41.9%**. On ZuCo the majority (positive) is 140 / 400 =
**35.0%**, so accuracy is a less misleading headline there.

Weighted F1 is the number I look at first because the original scripts
already print it.

## Gaze vs sentiment, linear

See [gaze-features.md](gaze-features.md) for the full correlation tables.

Takeaway I keep repeating to myself:

- Real ZuCo gaze is **weakly** related to the review label (|r| ≤ 0.07).
- Predicted full-SST gaze is also weak (|r| ≈ 0.05) and internally redundant.
- A model that only sees gaze should land near the majority floor unless
  there is a nonlinear pattern the correlations hide.

That is exactly what `examples/05_gaze_only_baseline.py` is for.

## Collinearity

On full SST, `nFix`, `FFD`, `TRT`, and `GPT` are interchangeable (r ≥ 0.99).
The fusion layer still has 5 inputs, but the effective rank is closer to 2
(`GD` vs “everything else”). If I ever retrain the predictor, I want it to
emit features that do not collapse.

On ZuCo, `nFixations`, `TRT`, and `GPT` travel together, but `SFD` and
`omissionRate` do not. The trainer drops those two. That may be leaving the
only complementary signal on the table.

## Length

| Corpus | Mean words | Max words | Tokenizer max |
| --- | ---: | ---: | ---: |
| ZuCo SST | 17.8 | 43 | 128 |
| Full SST | 19.2 | 56 | 128 |

Padding to 128 is convenient and wasteful. I have not seen a sentence get
truncated in these CSVs.

## Subject 3 hole

`ZuCo_et_csv_data/3_SR.csv` has 299 sentences. The other 11 readers have 400.
Sentence-level averages still have 400 rows, so those 101 missing sentences
are averaged over 11 readers instead of 12. Word averages are worse: they
groupby by **row index**, so the tail of `word_averages_v2.csv` is not a
clean 12-reader mean.

If I re-average later, I will join on `Sent_ID` + `Word_ID`, not on the
default RangeIndex.

## Split mismatch

`ZuCo_SST_data/train.csv` exists, but the ZuCo trainer never loads it. The
80/10/10 valid/test slices are 40 rows each and are too small to trust as a
single number (valid is 7 / 14 / 19 by class). Prefer the 5-fold protocol
that is actually in `model_ZuCo_SST.py`.

Full SST 80/10/10 is large enough. The seed is 42. Class shares stay similar
across splits; nobody accidentally dumped all neutrals into test.

## Known scoring bug on full SST test

Until `model_full_SST.py` accumulates predictions across batches, any printed
“Test Acc” is the last batch of 256 or fewer rows. Do not write that number
down as the test set. Valid metrics inside the epoch loop `extend()` correctly.

## What I want out of a future GPU run

1. `roberta` vs `roberta_eye_tracking` on ZuCo 5-fold, same seed, three
   repeats if the GPU is free.
2. The same pair on full SST, but with the test loop fixed and the
   checkpoint selected by valid F1 (the comment already thinks it does this).
3. A gaze-only logistic number next to both, so I can see whether fusion
   beats “RoBERTa plus a feature that a linear model cannot use.”

Until those exist, the honest claim of this repo is: **the plumbing to fuse
sentence-level gaze into BERT/RoBERTa is implemented, and the tables needed
to run it are checked in.**
