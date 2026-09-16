# Eye-tracking features

The classifiers fuse **five** sentence-level numbers. ZuCo tables store a
few more that are useful for analysis but unused at train time.

Example: `examples/03_gaze_feature_stats.py` prints per-class means,
feature–feature Pearson correlations, and a one-way ANOVA F-statistic
of each feature against `sentiment_label`.

## The five features in the fusion vector

Order in `model_full_SST.py`:

```text
['nFix', 'FFD', 'GPT', 'TRT', 'GD']
```

Order in `model_ZuCo_SST.py`:

```text
['nFixations', 'FFD', 'GPT', 'TRT', 'GD']
```

Same five quantities; ZuCo uses the longer fixation-count name.

### nFix / nFixations

**Number of fixations** that landed on the unit (word or, after
averaging, sentence). Higher values mean the eyes returned more often
or the unit took more samples to encode. At sentence level in
`DataTransformer`, word-level fixation counts are summed and then
divided by the number of words that had any fixation at all.

### FFD — first fixation duration

Duration of the **first** fixation on the word, in milliseconds in the
raw ZuCo CSVs. Isolated early-stage difficulty: unexpected or
emotionally loaded words often attract a longer first look.

### GD — gaze duration (first-pass time)

Sum of fixation durations on the word **before** the eyes leave it to
the right. First-pass integration, not including later regressions.

### TRT — total reading time

Sum of **all** fixation durations on the word, including returns from
the right. If a reader comes back after a sentiment flip at the end of
the sentence, TRT grows while GD may stay ordinary.

### GPT — go-past time (regression-path duration)

Time from first entering the word until the eyes first move **past** it
to the right, including any leftward regressions launched from that
region. GPT is the feature most sensitive to "I have to go back and
re-read."

## Extra ZuCo columns (not fused)

| Column | Meaning |
| --- | --- |
| `SFD` | Single fixation duration — GD when the word received exactly one fixation. |
| `omissionRate` | Sentence-level skip rate from ZuCo (`sent.omissionRate`). |
| `meanPupilSize` | Pupil size while fixating. Often used as a load / arousal proxy. |
| `SentLen` | Word count of the sentence. |
| `WordLen` | Character length of the token (word-level tables). |

## How sentence-level numbers are built

`DataTransformer` (`utils_ZuCo.py`), `level='sentence'`:

1. Walk each ZuCo `sentenceData` record and each `word`.
2. For every timing field, take the scalar attribute if it exists and is
   not an `ndarray`; otherwise contribute 0.
3. Accumulate those word values into a sentence vector.
4. Divide the timing / fixation fields by `nwords_fixated` (words that
   were not entirely zero). `SentLen` and `omissionRate` stay as-is.
5. Drop rows that contain `±inf`.
6. Optionally min-max, mean-normalize, or z-score **across sentences
   for that subject**.
7. Fill remaining NaNs with zeros, the feature min, or the feature mean.

A few subject/task combinations have corrupted sentence ranges; those
indices are hard-skipped (subject 2 on task 1 drops sentences 150–249
and 399, and so on). Only task 1 sentence CSVs are checked into
`ZuCo_et_csv_data/`.

Subject-averaged tables (`average_data.csv` and the scaled variants)
replace zeros with NaN **before** averaging so a skipped word in one
reader does not drag the mean to 0, then fill after scaling.

## Scaling you will actually see

| Table | Scaling |
| --- | --- |
| `ZuCo_et_csv_data/{1–12}_SR.csv` | Raw ms / counts (`scaling='raw'`) |
| `ZuCo_et_csv_data/average_data.csv` | Raw 12-subject mean |
| `ZuCo_et_csv_data/standard_scaled_average_data.csv` | `StandardScaler` on the average |
| `ZuCo_SST_data/combined_sst_et_standard.csv` | Same idea, already joined to text |
| `ZuCo_SST_data/combined_sst_et_min_max.csv` | Min-max on the average |
| `SST_data/*_full_sst.csv` | Already z-scored over all 11,853 rows |

Never mix raw ZuCo milliseconds with the full-SST z-scores in one
tensor. The fusion layer is a linear map — it can absorb a global
scale, but training will be miserable if a batch contains both.

## What the features are *not*

- They are not pixel coordinates or scanpath sequences. There is no
  spatial map of the page.
- They are not EEG bands. ZuCo recorded EEG; this repo does not load it.
- Predicted columns in `SST_data/combined_full_sst_et.csv` are not
  twelve new human subjects. Do not report them as "ZuCo on full SST".

## Why these five (and not pupil / SFD)

The fusion hidden layer is only 16 units. Five standard reading-time
features are the usual psycholinguistic baseline and match the
word-level prediction schema (`nFix`, `FFD`, `GPT`, `TRT`, `GD`). Pupil
size is a different construct (autonomic arousal). SFD is undefined
when a word has 0 or >1 fixations, so it is a sparse, awkward sentence
average. Both are left for the analysis scripts rather than the
classifier.

If you want to experiment, the place to add a sixth channel is
`num_eye_tracking_features` plus the column list in `load_dataset` /
the ZuCo training script. `examples/06_feature_fusion_walkthrough.py`
accepts `--num-gaze-features` so you can see the new concat width
before touching PyTorch.
