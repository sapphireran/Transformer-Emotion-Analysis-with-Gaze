# Gaze feature glossary

Sentence-level numbers in this repo are **means over fixated words**, not sums over the sentence. `DataTransformer` accumulates word features and then divides by `nwords_fixated`. Omitted words (no fixation) do not enter the mean, but they do enter `omissionRate`.

All millisecond measures below are the standard eyetracking definitions used in ZuCo-style exports. The tracker itself records raw samples; the `.mat` already stores per-word aggregates.

## The five fusion channels

### nFixations (nFix)

Number of discrete fixations that landed on the word, then averaged across fixated words in the sentence.

- **Goes up with:** longer words, low-frequency words, garden paths, rereading.
- **Confound:** longer sentences have more opportunities to reread; the per-word mean dampens but does not remove that.
- **In fusion:** a coarse "this sentence was looked at a lot" signal.

### FFD — First Fixation Duration

Duration of the **first** fixation on the word, ignoring later refixations.

- Classic early measure. Sensitive to lexical frequency and predictability.
- Less sensitive to later integration / sentiment wrap-up than GPT or TRT.
- After sentence averaging, FFD is a mild "average lexical difficulty" probe.

### GD — Gaze Duration (first-pass time)

Sum of all fixations on the word **during the first pass**, before the eyes leave it to the right.

- Includes immediate refixations (e.g. landing too far left on a long word).
- Still mostly an early/mid measure.
- Strongly correlated with FFD when most words are single-fixated; diverges on long words.

### TRT — Total Reading Time

Sum of **all** fixation durations on the word, including regressions back from the right.

- Late measure. Picks up reanalysis and wrap-up.
- In sentiment reviews, a spike can mean a twist, a negation, or just a hard name.
- Almost always the most correlated channel with nFixations (see `result/train_data_scatter_hist_plots.png`).

### GPT — Go-Past Time (regression-path duration)

Time from first entering the word until the eyes first move **past it to the right**, including any leftward regressions in between.

- The most "reanalysis-ish" of the five.
- Heavy-tailed. A few words with a long leftward excursion dominate a sentence mean.
- Useful, but brittle under min-max scaling: one outlier sets the ceiling.

## Extra channels sitting in the CSVs

### SFD — Single Fixation Duration

FFD restricted to words that were fixated exactly once. Cleaner early measure, fewer rows. Not in the fusion vector.

### omissionRate

`1 - (fixated words / sentence words)` at the sentence level, taken from ZuCo's `sent.omissionRate` when present, otherwise implied by zero word features.

High omission on short function words is normal. High omission on content words is a quality flag.

### meanPupilSize

Mean pupil size during fixations. In principle a load / luminance / arousal mix. In practice, luminance of the display and baseline pupil differ by subject, so the **standardized** table is the only one where this column is comparable across rows. Still not in the fusion vector.

### SentLen / WordLen

Length controls. Any claim that "gaze predicts sentiment" needs a length ablation. `examples/scripts/03_gaze_only_baseline.py` fits both a 5-channel model and a length-only model for that reason.

## Early vs late, in one picture

```
time -->

  FFD          GD              GPT                    TRT
  |            |               |                      |
  first        first-pass      first-pass             all visits
  fixation     including       + regressions          including
               refixations     before leaving         later rereads
```

For sentence-level sentiment, late measures (GPT, TRT) are the ones that *could* reflect wrap-up or affective re-reading. Early measures are closer to lexical difficulty. If a gaze-only classifier only wins with FFD and dies when you residualize on word frequency, the effect is probably not "emotion in the eyes."

## How sentence means are built

From `utils_ZuCo.py`, sentence branch, per word:

```python
word_features = [
    getattr(word, field) if hasattr(word, field)
    and not isinstance(getattr(word, field), np.ndarray)
    else 0
    for field in fields[2:]
]
features[idx, 2:] += word_features
nwords_fixated += 0 if (all zeros) else 1
# after the word loop:
features[idx, 2:] /= nwords_fixated
```

Missing attributes and array-valued attributes become 0. A word with all zeros does not increment `nwords_fixated`. Division by zero is possible if a sentence has no fixated words; those rows later go through `check_inf` / NaN fill.

## Subject averaging

`get_average_sentence_level.py` concatenates 12 subject frames and `groupby(level=0).mean()`. Zeros are first turned into NaN on every column except the id, so a skipped fixation does not drag the reader mean toward zero. That is different from `DataTransformer(fillna='zeros')`, which *does* write zeros before averaging if you average the raw exports yourself.

`word/get_average.py` currently **does not** replace 0 with NaN (the line is commented out). Word-level means therefore treat "no fixation" as a literal zero. Interpret word-level nFixations accordingly.

## Predicted gaze is not the same object

`gaze_prediction/data/prediction_test_v2.csv` is a model output: same column names, different generative process. Typical failure modes:

- The predictor copies length / frequency and never saw the ZuCo reader.
- Tokenization drift vs SST punctuation (`-LRB-` / `-RRB-` in the full SST dump).
- Calibration: values sit on a 0–100-ish scale, while ZuCo fusion uses z-scores.

Do not mix `prediction_test_v2.csv` rows into `combined_sst_et_standard.csv` without a rescale.

## Practical checks before training

1. No NaNs in the five fusion columns (`examples/gazekit/features.missingness_report`).
2. TRT vs nFixations correlation is high but not 1.0 — if it is 1.0, you are looking at a collapsed export.
3. Class-conditional means exist but are small relative to within-class variance. That is expected; gaze is a weak sentence-level sentiment feature.
4. Subject 3's shorter table did not shift `sentence_id` alignment (ids still 0..399 in the averaged files).
