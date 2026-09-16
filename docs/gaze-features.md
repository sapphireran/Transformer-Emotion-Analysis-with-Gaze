# Gaze features

Sentence- and word-level eye-tracking fields used in this repo. Units in the
**raw** ZuCo CSVs are counts, milliseconds, or arbitrary pupil-size units
from the tracker. Scaled tables are unitless.

The fusion head only sees five of these. The rest are kept for analysis and
for the averaging scripts.

## Feature glossary

### `nFixations` / `nFix`

Number of fixations landing on the word. At **sentence** level, ZuCo export
averages this across words that had at least one reported fixation (see
`nwords_fixated` in `DataTransformer`). High values: re-reading, difficulty,
or a long sentence with many content words.

### `FFD` — first fixation duration

Duration of the **first** fixation on the word. Classic early measure:
lexical access, frequency, predictability. In fusion tables this is averaged
up to the sentence.

### `SFD` — single fixation duration

Duration when the word received **exactly one** fixation. If the word was
skipped or fixated more than once, SFD is typically 0 / missing in the MATLAB
export, which becomes 0 after `fillna='zeros'`. That is why sentence-level
`SFD` is a noisy average and why it was **left out** of the five-D train
vector.

### `GD` — gaze duration (first-pass time)

Sum of fixations on the word during the **first pass**, before the eyes leave
it to the right. Includes refixations in that first visit, unlike FFD.

### `TRT` — total reading time

Sum of **all** fixations on the word, including later regressions. TRT ≥ GD
on a word (up to tracker noise). A large TRT − GD gap is a late, integration
cost signal.

### `GPT` — go-past time (regression-path duration)

Time from first entering the word until the eyes move to the **right** of it.
Includes regressions to earlier words. Often the largest of {FFD, GD, TRT,
GPT} when the reader bounces left to resolve negation or attachment.

### `meanPupilSize`

Mean pupil size over fixations on that word / sentence. Arousal, luminance,
and load all move it. Useful as a covariate; not in the fusion vector.

### `omissionRate`

Sentence-level fraction of words **never fixated**. High omission: skimming
or very predictable function words. Not in the fusion vector.

### `SentLen` / `WordLen`

Sentence length in tokens (from `len(sent.word)`) and character length of the
cleaned token. `WordLen` is kept on word tables only.

### `fixProp` (PROVO sample only)

Estimated probability that a word is fixated. Appears in
`gaze_prediction/data/provo.csv` **instead of** `GD`. Do not concatenate
PROVO rows with SST predicted files without renaming.

## How sentence-level numbers are built

From `utils_ZuCo.py` (sentence branch):

1. For each word, take the scalar gaze fields `nFixations, meanPupilSize, GD,
   TRT, FFD, SFD, GPT`. Missing or array-valued fields become 0.
2. Sum those into the sentence accumulator.
3. Count `nwords_fixated`: a word contributes if its feature vector is not
   `{0}`.
4. Divide the summed gaze fields by `nwords_fixated` (not by `SentLen`).
5. Set `omissionRate` from the sentence struct, `SentLen` from the word list
   length.

So sentence `nFixations` is **mean fixations per fixated word**, not total
fixations in the trial. Skipped words do not dilute the mean except via
`omissionRate`.

## Zeros vs missing

| Level | Typical meaning of 0 |
| --- | --- |
| Word gaze (`nFixations=0`, durations 0) | No fixation reported on that token |
| Word `SFD=0` with `nFixations≥2` | Not a single-fixation case |
| Sentence after `fillna='zeros'` | Could be true zero **or** filled missing |
| `get_average_sentence_level.py` | Converts 0 → NaN **before** the mean, then scaler |

Predicted full-SST columns are **never** raw milliseconds. They can be
negative after z-scoring (`train_full_sst.csv` has `nFix` values both below 0
and above 1).

## What the models actually consume

Hard-coded in both training scripts:

```python
num_eye_tracking_features = 5
# ZuCo human:
df[['nFixations', 'FFD', 'GPT', 'TRT', 'GD']]
# Full SST predicted:
df[['nFix', 'FFD', 'GPT', 'TRT', 'GD']]
```

Order is **nFix, FFD, GPT, TRT, GD** — early duration, regression-path,
total time, first-pass time. That mixes early and late measures on purpose.

Left out on purpose or by omission:

- `SFD` — mostly zero / collinear with FFD
- `omissionRate` — sentence skip rate; could be a 6th channel
- `meanPupilSize` — different unit and confound
- `SentLen` — the transformer already sees length via attention mask

## Scaling cheat sheet

| Table | Scaling |
| --- | --- |
| `ZuCo_et_csv_data/{1-12}_SR.csv` | Raw |
| `ZuCo_et_csv_data/average_data.csv` | Raw mean |
| `min_max_scaled_average_data.csv` | Min-max on the mean |
| `standard_scaled_average_data.csv` | Z-score on the mean |
| `combined_sst_et_min_max.csv` | Min-max gaze + text |
| `combined_sst_et_standard.csv` | Z-score gaze + text (**Track A train**) |
| `*_full_sst.csv` | Predicted, already scaled |
| `convert_zuco_data.py` output | Per-corpus min-max onto **0–100** |

Do not mix a raw CSV and a z-scored CSV in the same `DataLoader`. The linear
gaze layer (`5 → 16`) will happily run, but the weights are not comparable.

## Qualitative patterns worth checking

`examples/label_and_gaze_summary.py` prints class-conditional means on the
400-sentence z-scored table. Hypotheses that show up often in reading
research (not guaranteed here):

- Negative and sarcastic reviews: higher GPT / TRT (regressions on negation).
- Very positive adjectives: higher FFD on low-frequency praise words.
- Neutral / plot-summary lines: higher omission, lower nFixations.

The scatter-hist PNGs under `result/` are a visual check for the **predicted**
full-SST and PROVO distributions, not for ZuCo raw milliseconds.
