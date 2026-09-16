# File map

Paths are relative to the repository root. Scripts below are listed with the
**directory they were written to run from** when that matters (several use
bare filenames).

## Root training and conversion

| File | Reads | Writes / effect |
| --- | --- | --- |
| `utils_ZuCo.py` | `ZuCo_mat_data/{task}/*.mat` (12 files) | In-memory `pandas.DataFrame` per subject |
| `read_ZuCo_mat.py` | `DataTransformer('task1', level='sentence', scaling='raw')` | `et_csv_data/{1–12}_SR.csv` (**note:** folder name differs from `ZuCo_et_csv_data/`) |
| `get_average_sentence_level.py` | `et_csv_data/{1–12}_SR.csv` | `et_csv_data/min_max_scaled_average_data.csv`, `standard_scaled_average_data.csv` |
| `convert_full_SST.py` | `ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt` | `ZuCo_SST_data/ssts_ZuCo.csv` |
| `model_ZuCo_SST.py` | `ZuCo_SST_data/combined_sst_et_standard.csv` | Prints 5-fold metrics; no checkpoint |
| `model_full_SST.py` | `SST_data/{train,valid,test}_full_sst.csv` | `models/best_{model_type}_model.pth`, test metrics |

## `ZuCo_et_csv_data/` — human gaze, sentence level

400 rows per complete subject (header + 400 sentences). Subject `3_SR.csv` has
**300** data rows: ZuCo task 1, subject index 2, skips sentences 150–249 and
399 (see `DataTransformer.__call__`).

| File | Columns (sentence) |
| --- | --- |
| `{1–12}_SR.csv` | `id`, `SentLen`, `omissionRate`, `nFixations`, `meanPupilSize`, `GD`, `TRT`, `FFD`, `SFD`, `GPT` |
| `average_data.csv` | Same, mean across subjects (zeros treated as missing in `get_average_sentence_level.py`) |
| `min_max_scaled_average_data.csv` | Min-max of the averaged numeric columns |
| `standard_scaled_average_data.csv` | Z-score of the averaged numeric columns |

`id` here is the sentence index after the skip rules, **not** the raw MATLAB
loop index for the skipped subject.

## `ZuCo_et_csv_data/word/` — human gaze, word level

| File | Notes |
| --- | --- |
| `{1–12}_SR.csv` | One row per word. `Sent_ID` looks like `0_NR`. Subject 3 is shorter (`5293` words vs `7129`). |
| `word_averages.csv`, `word_averages_v2.csv` | Mean of gaze columns across subjects; `Word` / `WordLen` taken from subject 1 |
| `get_average.py` | Builds `word_averages_v2.csv` from the 12 word CSVs in the **same directory** |

Word columns: `id`, `Sent_ID`, `Word_ID`, `Word`, `nFixations`,
`meanPupilSize`, `GD`, `TRT`, `FFD`, `SFD`, `GPT`, `WordLen`.

Zeros in a word row usually mean **no fixation was reported** on that token
(skip / omission), not a literal 0 ms fixation.

## `ZuCo_SST_data/` — 400-sentence human-gaze classification set

| File | Role |
| --- | --- |
| `ssts_ZuCo.csv` | `sentence_id`, `sentence`, `sentiment_label` only |
| `combined_sst_et_min_max.csv` | Text + min-max gaze |
| `combined_sst_et_standard.csv` | Text + z-scored gaze (**this** is what `model_ZuCo_SST.py` loads) |
| `train.csv`, `valid.csv`, `test.csv` | 80 / 10 / 10 from `spilt.py` on the standard table (321 / 40 / 40) |
| `save_SST_data.py` | Alternate dump of `all/{label}/*.txt` → `output.csv` (run from this folder) |
| `spilt.py` | Filename is a typo of “split”; 80/10/10 with `random_state=42` |
| `ZuCo_SST_data.zip` | Archive of the same tables |

Joined gaze columns on the combined tables:

`omissionRate`, `nFixations`, `meanPupilSize`, `GD`, `TRT`, `FFD`, `SFD`, `GPT`

plus `sentence_id`, `sentence`, `sentiment_label`.

## `SST_data/` — full SST + predicted sentence gaze

| File | Role |
| --- | --- |
| `stts_all_sentence_level.csv` | Two columns: raw sentence, string label `POSITIVE` / `NEGATIVE` / `NEUTRAL` (no header) |
| `convert_sst_to_et.py` | Tokenize with NLTK, emit word rows with **zero** gaze placeholders → `sst_et_test.csv` |
| `sst_et_test.csv` | Word-level placeholder table (`nFix`…`GD` all 0) for the gaze predictor to fill |
| `combined_full_sst_et.csv` | Sentence-level text + predicted gaze |
| `spilt.py` | 80/10/10 → `train_full_sst.csv`, `valid_full_sst.csv`, `test_full_sst.csv` |
| `train_full_sst.csv` | ~9482 rows |
| `valid_full_sst.csv` | ~1185 rows |
| `test_full_sst.csv` | ~1186 rows |

Full-SST sentence columns:

`sentence_id`, `sentence`, `sentiment_label`, `nFix`, `GD`, `TRT`, `FFD`, `GPT`

Note the **`nFix` vs `nFixations`** rename relative to ZuCo tables.

## `gaze_prediction/data/`

| File | Role |
| --- | --- |
| `convert_zuco_data.py` | Min-max word gaze to 0–100; expects `training_data/word_averages_v2.csv` |
| `prediction_test.csv` | Small predicted-gaze sample (sentence_id starting at 300 in the checked-in file) |
| `prediction_test_v2.csv` | Large predicted word-level dump (~192k rows) aligned with SST tokenization |
| `provo.csv` | PROVO-style predicted rows; last numeric column is `fixProp`, not `GD` |

## `result/`

Scatter / histogram PNGs of train, test, and PROVO gaze-style features
(`*_scatter_hist_plots.png`). Useful for a qualitative check that predicted
values are in a plausible range; not generated by any script still in the
root.

## `examples/`

Read-only consumers of the CSVs above. They do not write training
checkpoints. See [../examples/README.md](../examples/README.md).
