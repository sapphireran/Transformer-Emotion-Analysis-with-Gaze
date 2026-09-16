# Eye-tracking features

The classifiers never see the raw gaze stream (x/y samples at 500 Hz).
They see a handful of **word- or sentence-aggregated reading-time
measures** that psycholinguistics has used for decades. This note defines
those measures, how this repo aggregates them, and which columns actually
reach the model.

## Measures

Times are milliseconds on a single word unless noted. After
cross-subject averaging and scaling they are no longer milliseconds —
see [Scaling](#scaling).

| Abbreviation | Full name | When it is measured | What it is usually taken to reflect |
|---|---|---|---|
| **nFix** / **nFixations** | Number of fixations | Count of fixations that landed on the word (or the mean count per fixated word, at sentence level) | How often the eyes returned to or lingered on the region |
| **FFD** | First-fixation duration | Duration of the *first* fixation in the region during first-pass reading | Early lexical access / surprise |
| **SFD** | Single-fixation duration | FFD in the special case that the word received exactly one fixation | Early processing when the word was read in one look |
| **GD** | Gaze duration (first-pass time) | Sum of fixations on the word from first entry until the eyes leave it to the *right* | Full first-pass recognition, including refixations |
| **GPT** | Go-past time (regression-path duration) | Time from first entry until the eyes move *past* the word to the right, including regressions to earlier words | Integration difficulty; the reader had to look back |
| **TRT** | Total reading time | Sum of *all* fixations on the word, including later passes | Overall attention allocated to the region |
| **meanPupilSize** | Mean pupil size | Average pupil diameter over fixations on the word | Arousal / load (noisy; lighting-sensitive) |
| **omissionRate** | Omission rate | Fraction of words in the sentence that received no fixation | Skipping; related to word length and predictability |
| **SentLen** | Sentence length | Word count of the sentence | Not an ET measure; used as a covariate in the raw tables |
| **WordLen** | Word length | Character count of the token | Strong predictor of nFix and skipping |
| **fixProp** | Fixation probability | Percent of readers who fixated the word (PROVO) | Skipping rate, inverted |

Only the five-feature subset `{nFixations, FFD, GPT, TRT, GD}` is
concatenated onto the transformer hidden state. `SFD`, `meanPupilSize`,
and `omissionRate` stay in the CSVs for analysis.

## Sentence-level aggregation (`DataTransformer`, `level='sentence'`)

For each sentence and each subject, `utils_ZuCo.py`:

1. Walks every word in `sentenceData[i].word`.
2. Reads the word-level attributes listed above (missing / array-valued
   attributes become `0`).
3. **Sums** `nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT`.
4. Divides those sums by the number of words that had any non-zero
   feature (words that were actually fixated).
5. Stores `SentLen = len(sent.word)` and the trial's `omissionRate`.

So a sentence-level `TRT` in `{k}_SR.csv` is “mean total reading time
per fixated word for this reader”, not the sum over the whole sentence.

Known-bad ZuCo trials are skipped before this aggregation. The important
Task 1 hole is subject index `2` (file `3_SR.csv`): sentences `150–249`
and `399` are dropped, which is why that file has 299 rows instead of
400.

## Word-level rows

At `level='word'` the transformer is not used. Each row is one token:

- `Sent_ID` = `"{sentence_index}_NR"` for Task 1/2
- `Word_ID` = 0-based position
- `Word` = token with leading punctuation stripped; the first token of
  the sentence is lowercased
- the seven ET columns plus `WordLen`

`ZuCo_et_csv_data/word/get_average.py` averages the seven ET columns
across the twelve subjects (row-aligned) and keeps identity columns from
subject 1.

## Cross-subject average

`get_average_sentence_level.py` concatenates the twelve sentence-level
files, replaces `0` with `NaN` in every column except `id`, then takes
the mean. That is why a missing subject does not drag the average toward
zero for that sentence.

Word-level averaging in `get_average.py` does **not** replace zeros with
NaN (the replacement is commented out), so unfixated words contribute
`0` to the mean.

## Scaling

Three feature-scaling modes exist in `DataTransformer`:

| Mode | Transform |
|---|---|
| `raw` | leave milliseconds / counts as recorded |
| `min-max` | `(x - min) / (max - min)` per column |
| `mean-norm` | `(x - mean) / (max - min)` |
| `standard` | `(x - mean) / std` |

The checked-in joins use sklearn's `MinMaxScaler` and `StandardScaler`
on the **already averaged** sentence table, which is equivalent to the
`min-max` / `standard` modes above applied once after averaging, not
per subject.

`gaze_prediction/data/convert_zuco_data.py` uses a third convention:
min-max onto `[0, 100]`, with `nFix` scaled on its own min/max and the
four duration features sharing one min/max. That is why predicted-gaze
files look like percentages, not z-scores.

Full-SST sentence tables (`train_full_sst.csv` etc.) contain **negative**
`nFix`/`GD`/… values. Those columns are already centered; do not treat
them as durations.

## Which columns the model sees

```python
# model_ZuCo_SST.py
eye_tracking_features = df[['nFixations', 'FFD', 'GPT', 'TRT', 'GD']]

# model_full_SST.py
eye_tracking_features = df[['nFix', 'FFD', 'GPT', 'TRT', 'GD']]
```

Order is **not** the same (`nFixations, FFD, GPT, TRT, GD` vs
`nFix, FFD, GPT, TRT, GD`). Both are 5-d, but a weight file trained on
one order is not interchangeable with the other if you ever swap CSVs.

## Exploratory plots

`result/` has three pairwise scatter + histogram grids:

| File | Distribution |
|---|---|
| `train_data_scatter_hist_plots.png` | predicted word-level gaze used as “train” |
| `test_data_scatter_hist_plots.png` | predicted word-level gaze used as “test” |
| `provo_data_scatter_hist_plots.png` | PROVO human norms |

The plots show the usual reading-time facts: `nFix` is right-skewed,
`TRT` tracks `nFix` almost linearly, `GD` tracks `FFD` on the lower
triangle (GD ≥ FFD), and `GPT` has a long right tail from regressions.

`examples/gaze_feature_report.py` prints per-class means and the
Pearson correlation of each sentence-level feature with the 0/1/2 label
so you can see how much linear signal the ET vector carries **without**
a transformer.
