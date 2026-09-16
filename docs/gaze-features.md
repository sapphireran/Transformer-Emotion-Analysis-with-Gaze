# Gaze features

Eye-tracking columns in this repo are standard reading-research measures.
They are **not** sentiment scores. They describe how a reader’s eyes
moved on a word or, after aggregation, on a sentence.

Sentence-level ZuCo tables keep eight measures plus sentence length:

```text
omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
```

Full-SST tables keep a five-feature subset under slightly shorter names:

```text
nFix, GD, TRT, FFD, GPT
```

`model_ZuCo_SST.py` and `model_full_SST.py` both feed **five** numbers into
the fusion MLP. The ZuCo trainer picks
`nFixations, FFD, GPT, TRT, GD` and ignores omission, pupil, and SFD.
The full-SST trainer picks the five columns that exist on that table.

---

## Per-word definitions

Take one token \(w\) in left-to-right order.

### nFixations / nFix

Count of discrete fixations whose landing position is on \(w\), including
later regressions. A skipped word has `nFixations = 0` and usually zeros
in every duration column.

On sentence-level ZuCo CSVs this is already **normalized by the number of
words that received at least one fixation** (`utils_ZuCo.py` divides the
sum of word features by `nwords_fixated`). So a sentence value of `2.2`
means “about 2.2 fixations per fixated word”, not “2.2 fixations on the
whole sentence”.

### FFD — first fixation duration

Duration of the **first** fixation on \(w\), in milliseconds in the raw
subject files. If the word was skipped, FFD is 0.

FFD is the usual first-pass “early” measure: it reacts to lexical
frequency and predictability more than to sentence-final wrap-up.

### SFD — single fixation duration

Duration of the only fixation on \(w\) when the word was fixated exactly
once. If the word received two or more fixations, SFD is 0 in the ZuCo
extract used here (`getattr` falling back to 0 when the field is missing
or is an array).

SFD is therefore sparse. That is why the full-SST projected table drops
it: a regressor that sees mostly zeros on multi-fixation words is a poor
transfer target.

### GD — gaze duration (first-pass time)

Sum of fixation durations on \(w\) during the **first pass**: from first
entering the word until the eyes leave it to the right, not counting
later regressions. Also called first-pass dwell time.

GD ≥ FFD when the word was fixated. GD = FFD when the first pass was a
single fixation.

### TRT — total reading time

Sum of **all** fixation durations on \(w\), first pass and later visits.
TRT ≥ GD. A large `TRT - GD` gap is a signature of regressions back onto
the word.

### GPT — go-past time (regression-path duration)

Clock time from first entering \(w\) until the eyes move to the *right*
of \(w\), including any regressions to earlier words in between.

GPT ≥ GD. GPT is the “this region caused trouble” measure: a garden-path
or an unexpected sentiment turn often shows up here even if first-pass GD
looked ordinary.

### meanPupilSize

Mean pupil size over fixations on the word, in the tracker’s pupil units
(ZuCo uses EyeLink). Larger pupils track luminance and also cognitive
load / arousal. It is **not** a sentiment label. Lighting and baseline
drift dominate, which is why some pipelines z-score per subject before
any cross-subject mean. The checked-in averages mix subjects first, then
scale.

### WordLen / SentLen

Character length of the cleaned token, or word count of the sentence.
Length correlates with almost every duration measure. If you compare
models “with gaze” vs “without gaze”, part of the gain can be a soft
length feature leaking through `nFixations` and `GD`. The toy example
prints a length-only baseline for that reason.

### omissionRate (sentence only)

ZuCo’s sentence-level skip rate: fraction of words with no fixation.
High omission on a long review is normal (short function words are
skipped). Extremely low omission on a five-word sentence is also normal.

---

## How sentence-level numbers are built

`DataTransformer` in `utils_ZuCo.py` (level `'sentence'`):

1. For each word, read the raw fields
   `nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT`.
2. Missing / array-valued fields become 0.
3. Sum those seven fields across words.
4. Divide the sum by `nwords_fixated` (words whose feature set is not
   the all-zero skip).
5. Store `SentLen = len(sent.word)` and `omissionRate` from the MATLAB
   sentence struct.
6. Optionally min-max / mean-normalize / z-score **per subject, per
   column**.
7. Fill remaining NaNs with zeros, column min, or column mean.

The checked-in `*_SR.csv` files were extracted with `scaling='raw'` and
`fillna='zeros'`, so they are still in tracker units. Cross-subject
means in `average_data.csv` are therefore in milliseconds / pupil units
/ counts. The experiment tables apply a **second** scale on the
already-averaged 400 × 8 matrix:

| File | Scale |
| --- | --- |
| `combined_sst_et_min_max.csv` | \((x - \min) / (\max - \min)\) per column |
| `combined_sst_et_standard.csv` | \((x - \mu) / \sigma\) per column |

`model_ZuCo_SST.py` reads the standard (z-score) table.

### What a skip looks like

Word-level subject file (`ZuCo_et_csv_data/word/1_SR.csv`):

```text
id,Sent_ID,Word_ID,Word,nFixations,meanPupilSize,GD,TRT,FFD,SFD,GPT,WordLen
2,0_NR,2,good,0,0,0,0,0,0,0,4
```

Subject 1 never fixated `good` in sentence 0. The averaged file
`word_averages_v2.csv` can still show a non-zero `nFixations` for that
row because other subjects did fixate it. That is the right average for
a “typical reader” feature. It is the wrong average if you wanted
skip-rate as its own signal — use `omissionRate` or count exact-zero
rows per subject instead (`examples/word_level_skip_analysis.py`).

---

## Typical raw magnitudes (subject-averaged, 400 sentences)

From `ZuCo_et_csv_data/average_data.csv` (raw units, not z-scored):

| Column | What “ordinary” looks like |
| --- | --- |
| `SentLen` | Roughly 5–30 words; SST snippets are short |
| `omissionRate` | Often 0.15–0.35 |
| `nFixations` | Around 1.5–3 fixations per fixated word |
| `meanPupilSize` | EyeLink pupil units, typically 800–900 in this extract |
| `FFD` | On the order of 100–140 ms |
| `SFD` | Lower than FFD; many zeros pulled the mean down |
| `GD` | A bit above FFD, often 120–180 ms |
| `TRT` | Above GD, often 170–360 ms |
| `GPT` | Above TRT more often than not on hard sentences |

Exact min / mean / max are printed by `examples/gaze_feature_stats.py`.

Z-scored experiment values in `combined_sst_et_standard.csv` are
**dimensionless**. A GPT of `6.09` on sentence 4 is “about six standard
deviations above the 400-sentence mean”, not six milliseconds.

Projected full-SST values in `train_full_sst.csv` are also not
milliseconds. Treat them as five extra floats the fusion layer is
allowed to see.

---

## Feature sets used by each program

| Program | Gaze vector |
| --- | --- |
| `model_ZuCo_SST.py` | `nFixations, FFD, GPT, TRT, GD` |
| `model_full_SST.py` | `nFix, FFD, GPT, TRT, GD` (same five, shorter names) |
| `examples/toy_text_gaze_fusion.py` | same five on the ZuCo z-score table |
| `examples/gaze_feature_stats.py` | all numeric gaze columns it finds |
| `utils_ZuCo.py` sentence extract | eight measures + `SentLen` |

The five-feature subset is the late-fusion contract: both trainers build
`nn.Linear(5, 16)`. If you add pupil or omission you must change
`num_eye_tracking_features` and the column list together.

---

## Correlations you should expect

On 400 real sentences, duration features are **positively correlated**.
GPT, TRT, and nFixations move together because they all grow when the
reader spends more time. FFD is the most independent of the early
measures; omissionRate often moves **opposite** nFixations (more skips →
fewer fixations per word after the script’s normalization).

That correlation is why a 16-d linear map over five inputs is enough:
the gaze head is not being asked to discover a 5-way interaction, only
to rotate a redundant duration cluster into the classifier’s space.

`examples/gaze_feature_stats.py` prints a Pearson matrix for the ZuCo
z-score table and for the full-SST projected table. If those two
matrices disagree violently, the projection model is not preserving the
reading-time geometry — worth knowing before you interpret a full-SST
ablation.

---

## Scaling choices

| Method | Flag in `DataTransformer` | Range | Use |
| --- | --- | --- | --- |
| Raw | `raw` | milliseconds, counts | Subject CSVs, inspection |
| Min-max | `min-max` | roughly `[0, 1]` per column | Bounded MLP inputs |
| Mean-normalize | `mean-norm` | roughly `[-1, 1]` | Rarely used here |
| Z-score | `standard` | mean 0, std 1 | Default ZuCo trainer |
| 0–100 display | `convert_zuco_data.py` | `[0, 100]` | Prediction-side CSV format |

Never mix scales inside one training run. The two `combined_sst_et_*.csv`
files are the same 400 rows under different transforms; they are not
additional data.

`examples/compare_scalings.py` checks that the min-max and z-score tables
share `sentence_id` / `sentence` / `sentiment_label` and only differ in
the numeric columns.

---

## What gaze is not

- Not a ground-truth emotion annotation.
- Not EEG. ZuCo recorded both; this clone kept ET.
- Not comparable across datasets without a convert step. PROVO’s
  `fixProp` (fixation probability, 0–100-ish) is not GD.
- Not safe to interpret on full SST as “readers looked here”. Those
  columns are model outputs.

A fair sentence for a paper draft: *we concatenate a learned projection
of five reading-time features with a transformer pooler*. A stretch:
*the model sees how people felt*. The second sentence is not supported
by these tables.
