# Gaze features

The fusion model does **not** take the whole ZuCo vector. Both training scripts index five columns, in this order when they build tensors:

1. `nFixations` (full SST: `nFix`)
2. `FFD`
3. `GPT`
4. `TRT`
5. `GD`

`EyeTrackingModel` then applies `nn.Linear(5, 16)` and concatenates that 16-d vector with the transformer pooler (768-d for `bert-base` / `roberta-base`).

## What each measure is

Times are milliseconds in the raw ZuCo extracts (`ZuCo_et_csv_data/average_data.csv`). After the standard join they are z-scores and no longer have units.

### nFixations / nFix

Count of fixations on the word. Sentence tables store the mean over words that were fixated (see `DataTransformer`: it divides the summed word features by `nwords_fixated`). Lots of fixations usually means refixation or a long word, not “the reader liked the movie.”

### FFD — First Fixation Duration

Length of the first fixation. Classic early measure. Short function words that get a single quick hit live here; rare or unexpected words run long.

### GD — Gaze Duration

First-pass dwell: all fixations on the word *before the eyes leave it*. Includes refixations in the first pass, excludes later rereading. In the raw sentence averages in this repo, GD sits between FFD and TRT (means about 141 / 117 / 203 ms for GD / FFD / TRT on `average_data.csv`).

### TRT — Total Reading Time

Every fixation on the word, including look-backs. Late, cumulative. Highly correlated with nFixations once you average to the sentence (the eyes cannot pile up time without piling up visits).

### GPT — Go-Past Time

Time from first entering a region until the eyes move *to the right of it*. Left-hand regressions count. This is the integration / “I have to go back” measure. On the raw sentence means it is the longest of the four durations (mean ≈ 242 ms) and the heaviest-tailed.

### Present in ZuCo extracts, unused by the fusion head

| Column | Why it exists | Why it is not in the 5-d vector |
|---|---|---|
| `SFD` | Single-fixation duration | Undefined on skipped or multiply-fixated words; lots of structural zeros |
| `meanPupilSize` | Load / lighting / arousal mix | Not a reading-time measure; different scale; easy to overfit 400 rows |
| `omissionRate` | Skip rate for the sentence | Already entangled with nFixations and sentence length |
| `SentLen` | Token count | The transformer already sees length through attention masks |
| `WordLen` | Characters (word tables only) | Lexical, not a gaze duration |

PROVO’s extra column `fixProp` is a *probability of fixation* (percent of readers), not a duration. It is only in `gaze_prediction/data/provo.csv`.

## Scaling you will actually see

| Table | Scaling | Consequence |
|---|---|---|
| `ZuCo_et_csv_data/{1-12}_SR.csv` | raw | milliseconds and counts, per reader |
| `ZuCo_et_csv_data/average_data.csv` | raw mean | comparable across sentences, not across features |
| `combined_sst_et_standard.csv` | z-score | this is what `model_ZuCo_SST.py` trains on |
| `combined_sst_et_min_max.csv` | [0, 1] | same rows, different numeric range |
| `SST_data/*_full_sst.csv` | predicted, already centered-ish | values typically land in roughly [-1, 2] |

Do not mix a raw table with a z-scored table in one `DataLoader`. The linear gaze layer will treat the scale as signal.

## Reader disagreement is part of the data

Subject 3 is missing 101 sentences. Pairwise Pearson correlations on raw `nFixations` (aligned by sentence id) are computed in `examples/02_gaze_feature_report.py`. Mean pairwise r on this checkout is **0.274** (best pair 0.453, worst 0.023). Twelve people do not read a review the same way, and the fusion model only ever sees the *average* reader.

## Feature glossary (generated from `tea_gaze`)

The helper package keeps the same descriptions so docs and code cannot drift:

```text
python -c "from tea_gaze.features import describe_features; print(describe_features())"
```
