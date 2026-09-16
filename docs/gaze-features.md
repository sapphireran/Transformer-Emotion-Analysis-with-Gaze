# Gaze features

Sentence-level eye-tracking is the extra signal this repo fuses into BERT / RoBERTa. This note defines the measures, how they are aggregated, and how they are scaled.

## Measure glossary

Times are durations. In the raw ZuCo export they are milliseconds. After min-max or standard scaling they are unitless.

| Code | Full name | Pass | What it usually reflects |
| --- | --- | --- | --- |
| `nFixations` / `nFix` | Number of fixations | all | How often the eyes landed on the region |
| `FFD` | First fixation duration | first | Early lexical access |
| `SFD` | Single fixation duration | first | Same as FFD when the word was fixated exactly once |
| `GD` | Gaze duration | first | Sum of first-pass fixations before leaving the word |
| `GPT` | Go-past time (regression path) | first + leftward | Time from first entry until the eyes move past the word to the right, including regressions |
| `TRT` | Total reading time | all | All fixations on the word, including later look-backs |
| `omissionRate` | Omission rate | sentence | Fraction of words with no fixation |
| `meanPupilSize` | Mean pupil size | all | Arousal / luminance confound; not used in the classifier |
| `SentLen` | Sentence length in words | — | From the ZuCo word array, not from the SST tokenizer |
| `WordLen` | Character length | word | Used in word-level tables only |

First-pass measures stop when the eyes leave the word to the right for the first time. `TRT` and extra fixations after that point are late measures.

## Sentence aggregation in `DataTransformer`

For each sentence and each numeric word field `f` in `{nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT}`:

1. Missing or array-valued fields become `0`.
2. Word values are summed.
3. The sum is divided by **the number of words that had at least one non-zero field**, not by `SentLen`.

Skipped words therefore do not enter the denominator. `omissionRate` is taken from the ZuCo sentence struct rather than recomputed. `SentLen` is `len(sent.word)`.

Word-level export does not average: each token keeps its own values. Empty tokens become `unknown` later in `get_average.py`.

## Subject averaging

Twelve readers produce twelve values per sentence (or per word). The committed averages are the arithmetic mean after replacing `0` with `NaN` on the sentence-level path. That treats a true zero fixation count and a missing recording the same way. If you need to distinguish "looked at the word for 0 ms" from "no data", regenerate from the `.mat` files and keep a separate missingness mask.

## Scaling conventions

`DataTransformer` can scale while exporting (`min-max`, `mean-norm`, `standard`, `raw`). The committed per-subject CSVs are **raw**. Scaling happens after the subject mean:

- **Min-max:** `(x - min) / (max - min)` per column, fit on the averaged table.
- **Standard:** `(x - mean) / std` per column, fit on the averaged table.
- **Mean-norm:** `(x - mean) / (max - min)` — implemented in the transformer, not used in the committed averages.

`gaze_prediction/data/convert_zuco_data.py` applies a different rule: `nFixations` is min-max scaled to `[0, 100]` on its own range, and `{FFD, GPT, TRT, GD}` share one min/max across those four columns. That is why predicted word-level files and ZuCo sentence-level files are **not** on the same numeric scale.

Full-SST sentence tables (`SST_data/*full_sst*.csv`) already contain scaled `nFix, GD, TRT, FFD, GPT`. Values can be negative; they behave like z-scores or another zero-centered transform, not like raw milliseconds.

## Which five columns enter the model

Both training scripts keep:

```text
nFixations or nFix, FFD, GPT, TRT, GD
```

Left out of the head:

- `SFD` — collinear with `FFD` on singly fixated words
- `omissionRate` — sentence-level skip rate; useful as a covariate in the example stats script
- `meanPupilSize` — lighting and individual differences dominate
- `SentLen` — the transformer already sees length through tokenization

If you add columns, bump `num_eye_tracking_features` **and** the `logits.view(-1, 3)` loss reshape is unrelated (that `3` is the label count).

## What to look at before training

`examples/gaze_feature_stats.py` reports, for each track:

- per-class means and standard deviations of the five fusion features
- Pearson correlation among those features
- a simple ANOVA-style F-statistic of feature vs. label (sklearn `f_classif`)
- omission-rate and sentence-length summaries on ZuCo rows

Typical patterns on this personal dump (re-check after any regenerate):

- `TRT` and `GPT` are strongly correlated; both capture re-reading.
- `FFD` and `GD` are moderately correlated first-pass pair.
- Extreme `nFixations` often sit on very short or very long sentences; inspect before you treat them as sentiment signal.
- Neutral reviews in the ZuCo slice are few. A feature that "separates neutral" may be a sample-size artifact.
- On the committed **full SST** table the five projected features are almost collinear (`|r|` often `> 0.98`). Class-conditional `f_classif` can still look significant because one latent direction moves slightly with the label. Treat that as a property of the projector, not as five independent reading-time measures.

## Leakage and interpretation

Gaze here is **human or predicted human** reading behavior, not a causal explanation of the label. A model that uses `TRT` is allowed to pick up "this sentence was hard to read," which can correlate with sarcasm, negation, or unusual vocabulary. Report text-only and text+gaze side by side. Do not claim the eyes caused the sentiment decision.
