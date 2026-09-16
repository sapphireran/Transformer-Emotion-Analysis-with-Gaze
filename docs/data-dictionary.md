# Data dictionary

Column names are not consistent across folders. This page is the join map.

## Sentiment columns

| Name | Files | Values |
| --- | --- | --- |
| `sentiment_label` | ZuCo combined, ZuCo splits, full SST fused | `0` negative, `1` neutral, `2` positive |
| (headerless 2nd col) | `SST_data/stts_all_sentence_level.csv` | `NEGATIVE` / `NEUTRAL` / `POSITIVE` |
| folder name | original `.txt` dump | same three strings |

`convert_full_SST.py` mapping:

```
NEGATIVE -> 0
NEUTRAL  -> 1
POSITIVE -> 2
```

## Text columns

| Name | Files | Notes |
| --- | --- | --- |
| `sentence` | all sentence-level modeling tables | Original review string, including punctuation |
| `Word` / `word` | word-level tables | Lowercased only when it was the first token of the sentence (`DataTransformer`) |
| `sentence_id` | fused tables | ZuCo: 0–399. Full SST: original corpus index |
| `id` | per-subject ET, averages | Same integer as ZuCo `sentence_id` |
| `Sent_ID` | word averages | `"{id}_NR"` on Task 1, `"{id}_TSR"` on Task 3 |
| `Word_ID` / `word_id` | word-level | 0-based index inside the sentence |

## Gaze channels used by the transformers

These five are the only ones fed to `EyeTrackingModel`:

| ZuCo name | Full SST name | Typical unit in raw ZuCo | Role in fusion |
| --- | --- | --- | --- |
| `nFixations` | `nFix` | count (mean per fixated word) | How often the eyes landed |
| `FFD` | `FFD` | ms | First fixation duration |
| `GPT` | `GPT` | ms | Go-past / regression-path time |
| `TRT` | `TRT` | ms | Total reading time |
| `GD` | `GD` | ms | Gaze duration / first-pass time |

`model_ZuCo_SST.py` selects `['nFixations', 'FFD', 'GPT', 'TRT', 'GD']`.
`model_full_SST.py` selects `['nFix', 'FFD', 'GPT', 'TRT', 'GD']`.

If you concatenate the two tables, rename before stacking. `examples/gazekit/schema.py` exposes `CANONICAL_GAZE = ("nFixations", "FFD", "GPT", "TRT", "GD")` and a rename map.

## Extra sentence-level ET (ZuCo only)

Present on ZuCo sentence tables, **not** passed into the classifier:

| Name | Meaning |
| --- | --- |
| `SentLen` | Number of word objects in the ZuCo sentence |
| `omissionRate` | Fraction of words with no recorded fixation |
| `meanPupilSize` | Mean pupil size over fixations (arbitrary eyetracker units) |
| `SFD` | Single-fixation duration |

These are useful covariates. A gaze-only baseline that includes `omissionRate` and `SentLen` is a stronger "difficulty" control than the five fusion channels alone.

## Word-level extras

| Name | Meaning |
| --- | --- |
| `WordLen` | `len(token)` after punctuation stripping |
| `fixProp` | PROVO table only: fixation probability (percent-like) |

## Scaling state

Always check which file you opened before interpreting magnitudes.

| File pattern | Scaling |
| --- | --- |
| `ZuCo_et_csv_data/{n}_SR.csv` | Raw-ish means in ms / counts (`DataTransformer(..., scaling='raw')`) |
| `average_data.csv` | Raw-ish reader mean |
| `standard_scaled_average_data.csv`, `combined_sst_et_standard.csv` | Column z-score |
| `min_max_scaled_average_data.csv`, `combined_sst_et_min_max.csv` | Column min-max to ~[0, 1] |
| `*_full_sst.csv` | Already scaled (roughly unit-ish, not a clean z-score) |
| `gaze_prediction/data/prediction_*.csv` | Predictor output scale (often 0–100 style) |
| `gaze_prediction/data/provo.csv` | PROVO / converted scale |

`convert_zuco_data.py` independently min-maxes `nFixations` and, **as a group**, `FFD/GPT/TRT/GD` onto `[0, 100]`. That is a third scaling, used for foreign plotters, not for `EyeTrackingModel`.

## Tokenization mismatch

Three tokenizers appear in this repo:

1. **ZuCo native words** — `word.content` inside the `.mat`, lightly stripped with `re.sub('[^\w\s]', '')`.
2. **NLTK `word_tokenize`** — `SST_data/convert_sst_to_et.py`, then alphabetic filter.
3. **BERT / RoBERTa WordPiece / BPE** — training scripts, `max_length=128`, `padding='max_length'`.

Gaze is never aligned to WordPiece ids. The model only sees a sentence-level vector. That avoids alignment bugs and also throws away word-level information that the CSVs still contain.

## Empty / placeholder tokens

- `DataTransformer` and `get_average.py` replace missing words with `unknown`.
- `convert_sst_to_et.py` writes a single `unknown` token when NLTK + the alpha filter produce no words.
- `convert_zuco_data.py` does the same when `Word` is empty.

Downstream counts of vocabulary size should drop `unknown`.

## File-format quirks

- `stts_all_sentence_level.csv` has **no header**. `pd.read_csv` will steal the first sentence as column names unless you pass `header=None`.
- Several ZuCo frames were saved with MultiIndex columns from `columns=[fields]` in `DataTransformer`. The checked-in CSVs look flat; if you regenerate from `.mat` you may need `df.columns = df.columns.get_level_values(0)`.
- `model_full_SST.py` comment says it saves the best model "with F1" but the predicate is `val_acc > best_val_acc`.
