# Eye-tracking features

All duration features in the raw ZuCo exports are in **milliseconds**.
Count features are dimensionless. After `get_average_sentence_level.py`
or `DataTransformer(scaling=...)`, the numbers you see in the combined
CSVs are **scaled**, not milliseconds.

## The five features the fusion head uses

`EyeTrackingModel` always takes a tensor of shape `(batch, 5)`. Column
order is whatever the training script selects:

```python
# model_ZuCo_SST.py
df[['nFixations', 'FFD', 'GPT', 'TRT', 'GD']]

# model_full_SST.py
df[['nFix', 'FFD', 'GPT', 'TRT', 'GD']]
```

Order is **not** the same as the CSV header order on the full-SST
files (`nFix, GD, TRT, FFD, GPT`). The `DataLoader` uses the explicit
column list above, so training is consistent. If you hand a raw CSV
row to the linear layer without reordering, you will silently permute
gaze dimensions.

### nFix / nFixations

Number of fixations on the unit (word, or average over fixated words
in the sentence). Higher nFix often means rereading or struggling with
the region. At sentence level in `DataTransformer`, word-level nFix
values are **summed then divided by the number of words that had any
fixation**, so this is closer to “mean fixations per fixated word”
than to a raw total.

### FFD — First Fixation Duration

Duration of the **first** fixation on the word. Classic early measure:
lexical access, surprise at the first encounter. Short function words
usually have smaller FFD than rare adjectives.

### GD — Gaze Duration (first-pass time)

Sum of fixations on the word from first entry until the eyes leave to
the right, **before** a regression back. Early-to-mid measure. If the
reader immediately bounces back from later words, those extra looks
are **not** in GD; they land in TRT / GPT.

### TRT — Total Reading Time

All fixation time on the word, including later passes. Late measure.
A sarcastic twist at the end of a review can inflate TRT on an earlier
adjective the reader returns to.

### GPT — Go-Past Time (regression-path duration)

Time from first entering the word until the eyes first move **past**
it to the right, including regressions to earlier words. GPT ≥ GD.
Large GPT − GD gaps are a useful “this region caused a regressive
loop” signal. The current model feeds both GPT and GD as raw (scaled)
inputs and never computes that difference explicitly.

## Features extracted but not fused

Present on ZuCo sentence tables, ignored by `get_model` / `CustomDataset`
beyond being in the CSV:

| Column | What it is | Why it is sitting unused |
| --- | --- | --- |
| `omissionRate` | Fraction of words never fixated | Strong proxy for skipping / skimming |
| `meanPupilSize` | Pupil diameter (ZuCo units) | Arousal / load; very subject-dependent |
| `SFD` | Single Fixation Duration | Only defined when nFix == 1 |
| `SentLen` | Word count | Trivial text feature; not gaze |
| `WordLen` | Character length (word tables) | Strongly correlated with nFix / TRT |

If you add any of these to the linear gaze layer, bump
`num_eye_tracking_features` and the column list together. The
classifier input size is `hidden_size + hidden_layer_size` (768 + 16)
and does not care how the 16-d vector was built — only the
`nn.Linear(n, 16)` does.

## Sentence-level aggregation (`DataTransformer`, level=`sentence`)

For each sentence and each duration/count field in
`[nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT]`:

1. Walk `sent.word`.
2. If the MATLAB struct has a scalar attribute, add it; if the
   attribute is missing or is an `ndarray`, add 0.
3. Count how many words had a non-all-zero feature vector
   (`nwords_fixated`).
4. Divide the sums by that count.
5. Store `SentLen = len(sent.word)` and `omissionRate` from the
   sentence struct.

So sentence-level FFD is “mean first-fixation duration over words that
were actually looked at,” not a sum. That is why a 5-word sentence and
a 40-word sentence can have similar FFD after aggregation.

Known-bad sentence index ranges are skipped per subject (Task 1
subject 2, Task 2 subjects 6 and 11, Task 3 subjects 3, 7, 11). Those
cuts are why `3_SR.csv` has 299 rows instead of 400.

## Scaling

`DataTransformer` supports `min-max`, `mean-norm`, `standard`, and
`raw`. `read_ZuCo_mat.py` writes **raw** per-subject files.

`get_average_sentence_level.py` then:

1. Replaces 0 with NaN on every column except the first (`id` / `SentLen`
   handling depends on the file).
2. Averages by row index across the 12 CSVs.
3. Fits `MinMaxScaler` and `StandardScaler` on the averaged table
   (id held out).

`model_ZuCo_SST.py` trains on the **standard**-scaled join. The
min–max join is an unused sibling file — handy for an ablation but
not wired up.

`gaze_prediction/data/convert_zuco_data.py` uses a different rule:
nFixations is scaled 0–100 from its own min/max; FFD/GPT/TRT/GD share
**one** joint min/max. That makes duration features comparable to each
other, and incomparable to the z-scored sentence tables.

## Missing data

| Stage | Policy |
| --- | --- |
| MATLAB attribute missing | 0 |
| `fillna='zeros'` (default) | NaN → 0 after scaling |
| Averaging script | 0 → NaN **before** mean, so a zero does not vote |
| Word averages v2 | `Word` NaN → `unknown`; numeric NaN → 0 |
| SST placeholder (`convert_sst_to_et.py`) | every gaze cell is 0 |

A row of five zeros after standard scaling is **not** “no information”;
it is whatever z-score a raw 0 mapped to, or a true zero if the
placeholder path was used. The toy fusion example prints how many
rows are exactly zero so you can see the difference.

## Correlations you should expect

These are psycholinguistic regularities, not results from a paper:

- `WordLen` ↑ → `nFixations`, `TRT`, `GD` ↑
- `FFD` and `GD` move together when most words are single-fixation
- `GPT` and `TRT` move together when readers regress
- `omissionRate` ↑ → fewer content-word hits, often shorter / easier
  or more skimmable sentences
- Sentiment is **weakly** related to any single gaze column. If late
  fusion helps, it is usually from the **combination** with the text
  encoding, not from “negative reviews have longer TRT” as a rule.

`examples/03_gaze_feature_summary.py` prints means by class and a
correlation matrix on the 400-sentence standard table so you can see
the actual numbers in this checkout.

## Naming mismatches (read before joining tables)

| Concept | ZuCo sentence CSV | Full SST CSV | Word average | PROVO helper |
| --- | --- | --- | --- | --- |
| fixation count | `nFixations` | `nFix` | `nFixations` | `nFix` |
| go-past | `GPT` | `GPT` | `GPT` | `GPT` |
| last extra column | `SFD` | — | `WordLen` | `fixProp` (not GD!) |

`provo.csv` does **not** have `GD`. Do not concatenate it with
`prediction_test.csv` without renaming.
