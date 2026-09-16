# Data pipeline

Everything below refers to files that are already in this personal checkout. MATLAB sources (`ZuCo_mat_data/`) are **not** in the repo; `read_ZuCo_mat.py` expects you to have dropped them next to the checkout if you want to regenerate the CSVs.

## 1. ZuCo MATLAB → per-reader sentence tables

`utils_ZuCo.get_matfiles()` looks under `ZuCo_mat_data/{task}` and asserts **12** `.mat` files (one reader each). The path string still uses Windows backslashes; on Linux you will want to change that if you re-run the extract.

`DataTransformer(task, level, scaling, fillna)` then walks `sentenceData`:

- **Task 1** is the normal-reading movie-review pass used here (`read_ZuCo_mat.py` hard-codes `task1`, `level='sentence'`, `scaling='raw'`).
- Several reader/task pairs skip known bad sentence ranges (the long `if` chain in `utils_ZuCo.py`). That is why `ZuCo_et_csv_data/3_SR.csv` has **299** sentences and the others have **400**.
- Sentence-level fields: `SentLen`, `omissionRate`, `nFixations`, `meanPupilSize`, `GD`, `TRT`, `FFD`, `SFD`, `GPT`.
- Word-level fields add `Sent_ID`, `Word_ID`, `Word`, `WordLen`. Task 1/2 ids look like `12_NR`; task 3 uses `_TSR`.

`read_ZuCo_mat.py` writes `et_csv_data/{i}_SR.csv`. The committed copies live in `ZuCo_et_csv_data/`.

## 2. Average readers, then scale

`get_average_sentence_level.py` (folder name in the script is still `et_csv_data`) treats zeros as missing, averages aligned rows across readers, and writes:

- `min_max_scaled_average_data.csv`
- `standard_scaled_average_data.csv`

The unscaled mean table in the repo is `ZuCo_et_csv_data/average_data.csv`.

Word-level counterpart: `ZuCo_et_csv_data/word/get_average.py` → `word_averages_v2.csv` (7129 data rows, 400 `*_NR` sentences).

## 3. Join Stanford sentiment labels

`ZuCo_SST_data/ssts_ZuCo.csv` is the 400-row text+label table (`0/1/2`). `convert_full_SST.py` and `ZuCo_SST_data/save_SST_data.py` both rebuild that file from a folder of `NEGATIVE/POSITIVE/NEUTRAL/*.txt` files that are **not** committed.

The joined products are:

| File | Gaze scaling | Rows |
|---|---|---|
| `ZuCo_SST_data/combined_sst_et_standard.csv` | z-score | 400 |
| `ZuCo_SST_data/combined_sst_et_min_max.csv` | min-max | 400 |

`model_ZuCo_SST.py` reads the **standard** join. `ZuCo_SST_data/spilt.py` then cuts an 80/10/10 hold-out (`train.csv` / `valid.csv` / `test.csv`). The transformer script **does not use that hold-out**; it re-splits with 5-fold CV. The hold-out is still useful for the sklearn examples.

Label mix on the 400-row join:

| Label | Name | Count |
|---:|---|---:|
| 0 | NEGATIVE | 123 |
| 1 | NEUTRAL | 137 |
| 2 | POSITIVE | 140 |

## 4. Full SST + predicted gaze

`SST_data/stts_all_sentence_level.csv` is the raw SST dump used as text (headerless `sentence,POLARITY` rows). `SST_data/convert_sst_to_et.py` tokenizes with NLTK and writes a word table whose gaze columns are zeros — a scaffold for the predictor.

`gaze_prediction/data/convert_zuco_data.py` rescales a word-average table into the predictor’s column layout (`nFix`, `FFD`, `GPT`, `TRT`, `GD`) on a 0–100 range.

Checked-in predictor outputs:

- `gaze_prediction/data/prediction_test.csv` — 1751 word rows, 100 sentences
- `gaze_prediction/data/prediction_test_v2.csv` — large word-level dump (same width as the SST word scaffold)
- `gaze_prediction/data/provo.csv` — 2659 PROVO word rows (134 sentences), with `fixProp` instead of `GD`

Sentence-level predicted gaze is already joined in:

| File | Rows |
|---|---:|
| `SST_data/combined_full_sst_et.csv` | 11853 |
| `SST_data/train_full_sst.csv` | 9482 |
| `SST_data/valid_full_sst.csv` | 1185 |
| `SST_data/test_full_sst.csv` | 1186 |

`SST_data/spilt.py` is the 80/10/10 cut (`random_state=42`). Columns: `sentence_id, sentence, sentiment_label, nFix, GD, TRT, FFD, GPT`.

Neutral is the minority class on the large split (train: 3710 / 1833 / 3939). That is a different prior than the almost-balanced ZuCo overlap.

## 5. Column aliases the trainers rely on

```text
ZuCo join:   nFixations, FFD, GPT, TRT, GD   (+ omissionRate, meanPupilSize, SFD)
Full SST:    nFix,       FFD, GPT, TRT, GD
```

`tea_gaze.features.canonicalize_gaze_columns()` treats `nFix` as `nFixations` so the examples can share one fusion-feature list.

## 6. Files you can ignore while reading docs

- `result/*.png` — older scatter/histogram dumps of predicted word-level features (nFix, FFD, GPT, TRT, GD). They are not produced by the new examples.
- Duplicate split helpers (`spilt.py` in two folders).
- `convert_full_SST.py` vs `ZuCo_SST_data/save_SST_data.py` — same idea, different output paths.

Regenerating any of this from MATLAB or from raw SST folders is optional. The examples only read the committed CSVs.
