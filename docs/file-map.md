# File map

Paths are relative to the repository root. Scripts are listed with the **paths they actually hard-code**, not the paths a cleaned-up rewrite would use.

## Training

| File | Role | Input | Output |
| --- | --- | --- | --- |
| `model_ZuCo_SST.py` | 5-fold CV, optional gaze concat | `ZuCo_SST_data/combined_sst_et_standard.csv` | printed fold metrics |
| `model_full_SST.py` | train / valid / test, optional gaze concat | `SST_data/{train,valid,test}_full_sst.csv` | `models/best_{model_type}_model.pth` plus printed metrics |

Both files duplicate `EyeTrackingModel`, `CustomDataset`, `get_model`, and `calculate_metrics`. Hyperparameters are constants at the top of each file (no CLI).

## ZuCo MATLAB ingest (requires files not in git)

| File | Role | Input | Output |
| --- | --- | --- | --- |
| `utils_ZuCo.py` | `get_matfiles`, `DataTransformer`, `split_data` | `os.getcwd() + '\\ZuCo_mat_data\\' + task` (Windows-style) | sentence- or word-level `pandas.DataFrame` |
| `read_ZuCo_mat.py` | dump Task 1 sentence tables | `DataTransformer('task1', level='sentence', scaling='raw')` | `et_csv_data/{1–12}_SR.csv` (**not** `ZuCo_et_csv_data/`) |

Committed sentence tables live in `ZuCo_et_csv_data/`. To re-run `read_ZuCo_mat.py` you must create `et_csv_data/` or edit the path.

## Averaging and joining

| File | Role | Input | Output |
| --- | --- | --- | --- |
| `get_average_sentence_level.py` | mean over 12 subject CSVs, then MinMax + Standard scale | `et_csv_data/{1–12}_SR.csv` | `et_csv_data/{min_max,standard}_scaled_average_data.csv` |
| `ZuCo_et_csv_data/word/get_average.py` | mean word-level gaze | `{1–12}_SR.csv` in the `word/` folder | `word_averages_v2.csv` |
| `convert_full_SST.py` | SST text files → `ssts_ZuCo.csv` | `ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt` | `ZuCo_SST_data/ssts_ZuCo.csv` |
| `ZuCo_SST_data/save_SST_data.py` | same mapping, different relative paths | `all/{...}/*.txt` | `output.csv` |

The 400-row `ssts_ZuCo.csv` **is** committed. The `all/` text-file tree is **not** (a zip `ZuCo_SST_data/ZuCo_SST_data.zip` is). Combined tables were produced by joining `ssts_ZuCo.csv` to scaled sentence gaze on `sentence_id == id` and dropping `SentLen`.

## Splits

| File | Input | Outputs |
| --- | --- | --- |
| `ZuCo_SST_data/spilt.py` | `combined_sst_et_standard.csv` | `train.csv`, `valid.csv`, `test.csv` (320 / 40 / 40) |
| `SST_data/spilt.py` | `combined_full_sst_et.csv` | `train_full_sst.csv`, `valid_full_sst.csv`, `test_full_sst.csv` (9482 / 1185 / 1186) |

Both use `sklearn.model_selection.train_test_split` twice with `random_state=42` and `test_size=0.2` then `0.5` on the remainder. Filename is `spilt.py` (typo).

## SST word-level placeholders and predicted gaze

| File | Role |
| --- | --- |
| `SST_data/convert_sst_to_et.py` | tokenize `stts_all_sentence_level.csv` with NLTK; write zeros for `nFix,FFD,GPT,TRT,GD` → `sst_et_test.csv` |
| `gaze_prediction/data/convert_zuco_data.py` | scale word-level ZuCo averages into the predicted-gaze CSV layout (0–100 min-max) |
| `gaze_prediction/data/prediction_test.csv` | 1,751 word rows, sentences 300–399 |
| `gaze_prediction/data/prediction_test_v2.csv` | 191,971 word rows (full SST tokenization) |
| `gaze_prediction/data/provo.csv` | Provo corpus word-level gaze (`fixProp` instead of `GD`) |

The **gaze-prediction model code is not in this repository** — only converted tables and the `result/*.png` diagnostics.

## Data directories

| Directory | What is inside |
| --- | --- |
| `ZuCo_et_csv_data/` | 12 subject sentence CSVs, `average_data.csv`, two scaled averages |
| `ZuCo_et_csv_data/word/` | 12 subject word CSVs plus `word_averages.csv` and `word_averages_v2.csv` |
| `ZuCo_SST_data/` | 400 labeled sentences, two joined scalings, 80/10/10 split, zip of source texts |
| `SST_data/` | raw SST sentences, combined predicted gaze, splits, word-level zero table |
| `gaze_prediction/data/` | predicted gaze + Provo + conversion helper |
| `result/` | scatter/histogram grids for train / test / Provo word-level features |
| `docs/` | this guide |
| `examples/` | CPU analysis package, scripts, tests |

## Examples package (added in this documentation work)

| Path | Role |
| --- | --- |
| `examples/teag_examples/` | importable helpers (`teag` = Transformer Emotion Analysis with Gaze) |
| `examples/scripts/` | command-line entry points |
| `examples/tests/` | pytest suite against the committed CSVs |
| `docs/assets/` | figures produced by those scripts |
