# Data dictionary

Column names are not uniform. The ZuCo trainers say `nFixations`; the full SST
trainers say `nFix`. Examples in `tea_gaze.schema` keep both lists instead of
renaming on load.

## Sentiment tables

| Column | Type | Used by | Meaning |
| --- | --- | --- | --- |
| `sentence_id` | int | joins | 0-based sentence index in that table |
| `sentence` | str | tokenizer | Review sentence, punctuation kept |
| `sentiment_label` | int `{0,1,2}` | loss | 0 negative, 1 neutral, 2 positive |

`SST_data/stts_all_sentence_level.csv` is the odd one out: no header, two
columns, labels as the strings `NEGATIVE` / `NEUTRAL` / `POSITIVE`.
`convert_full_SST.py` maps those strings with
`NEGATIVE=0, NEUTRAL=1, POSITIVE=2`.

## Sentence-level ZuCo eye tracking

Present on `ZuCo_et_csv_data/*_SR.csv`, the scaled averages, and the combined
ZuCo SST tables.

| Column | Typical range (raw average) | Meaning |
| --- | --- | --- |
| `id` | 0–399 | Sentence index in the reader export |
| `SentLen` | ~5–40 | Word count stored by `DataTransformer` |
| `omissionRate` | 0–1 raw | Share of words with no fixation |
| `nFixations` | ~1–4 raw | Mean fixation count on fixated words |
| `meanPupilSize` | ~800–900 raw | Mean pupil size |
| `GD` | milliseconds | Gaze duration / first pass |
| `TRT` | milliseconds | Total reading time |
| `FFD` | milliseconds | First fixation duration |
| `SFD` | milliseconds | Single fixation duration |
| `GPT` | milliseconds | Go-past / regression-path time |

On `combined_sst_et_standard.csv` those ET columns are z-scored (mean ≈ 0,
std ≈ 1). On `combined_sst_et_min_max.csv` they sit in `[0, 1]`.

`model_ZuCo_SST.py` only feeds five of them into the network:

```
nFixations, FFD, GPT, TRT, GD
```

`omissionRate`, `meanPupilSize`, and `SFD` are in the CSV and unused by that
trainer.

## Word-level ZuCo eye tracking

`ZuCo_et_csv_data/word/*_SR.csv` and `word_averages_v2.csv`.

| Column | Meaning |
| --- | --- |
| `id` | Row index in that subject file |
| `Sent_ID` | `{sentence_index}_NR` for normal reading (task 1) |
| `Word_ID` | Token index inside the sentence |
| `Word` | Lightly cleaned token; empty tokens become `unknown` in the average file |
| `nFixations` … `GPT` | Same measures as sentence level, per word |
| `WordLen` | `len(token)` after stripping punctuation |

A zero row usually means “this reader never fixated that word.” The sentence
transformer averages only over words that had a fixation (`nwords_fixated`).

## Full SST sentence table

`SST_data/combined_full_sst_et.csv` and the three splits.

| Column | Meaning |
| --- | --- |
| `sentence_id` | Index from the original SST dump |
| `sentence` | SST tokenized text (`-LRB-` / `-RRB-` style) |
| `sentiment_label` | Same 0/1/2 map |
| `nFix` | Predicted fixation count (z-scored in the combined file) |
| `GD` | Predicted gaze duration |
| `TRT` | Predicted total reading time |
| `FFD` | Predicted first fixation duration |
| `GPT` | Predicted go-past time |

`model_full_SST.py` reads those five ET columns in the order
`nFix, FFD, GPT, TRT, GD`.

## Word skeleton and predicted-word tables

`SST_data/sst_et_test.csv` and `gaze_prediction/data/prediction_test*.csv`:

| Column | Meaning |
| --- | --- |
| `sentence_id` | SST sentence index |
| `word_id` | Token index after `nltk.word_tokenize` + alphabetic filter |
| `word` | Token string |
| `nFix`, `FFD`, `GPT`, `TRT`, `GD` | Zeros in the skeleton; predicted values later |

`gaze_prediction/data/provo.csv` swaps `GD` for `fixProp` (fixation
proportion). Do not concat it onto the ZuCo trainers without renaming.

## Label map used everywhere in this repo

```
NEGATIVE -> 0
NEUTRAL  -> 1
POSITIVE -> 2
```

That map is hard-coded in `convert_full_SST.py` and `ZuCo_SST_data/save_SST_data.py`.
The trainers never see the string names.

## Checks the example loaders run

`tea_gaze.io` refuses to load a table if the schema kind is missing columns:

- `zuco_text`: `sentence_id, sentence, sentiment_label`
- `zuco_sentence_et`: text columns + the eight ZuCo ET columns
- `full_sst`: text columns + `nFix, FFD, GPT, TRT, GD`
- `zuco_word`: `id, Sent_ID, Word_ID, Word` + word ET columns

Subject files are loaded more loosely because subject 3 has fewer rows and the
raw exports still use `id` instead of `sentence_id`.
