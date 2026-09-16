# Data pipeline

The repo stores **derived tables**, not the original ZuCo MATLAB dumps or the raw SST tree files. This note walks the scripts that produced those tables so you can regenerate or extend them.

## Sources

### ZuCo

ZuCo records simultaneous EEG and eye-tracking while twelve participants read English sentences. Task 1 in this repo is the sentiment-reading (`SR`) subset: movie-review sentences that overlap SST. `utils_ZuCo.DataTransformer` expects twelve `.mat` files per task under `ZuCo_mat_data/task1` (path separator is currently Windows-style).

Each subject file exposes `sentenceData`, and each sentence exposes a `word` array with per-word eye-tracking fields. The transformer walks those structs and writes either:

- **sentence level** — one row per sentence, word measures averaged over fixated words
- **word level** — one row per word, plus `Sent_ID`, `Word_ID`, `Word`, `WordLen`

A handful of subject / sentence ranges are skipped because those recordings are empty or corrupt in the public release. The skip lists are hardcoded in `DataTransformer.__call__`.

### SST

Two SST views appear in the repo:

1. **ZuCo-aligned SST** — the ~400 review sentences that ZuCo actually recorded. Built from a folder of `.txt` files (`convert_full_SST.py`, `ZuCo_SST_data/save_SST_data.py`) and saved as `ZuCo_SST_data/ssts_ZuCo.csv`.
2. **Full SST** — the larger sentence-level dump in `SST_data/stts_all_sentence_level.csv` (`sentence`, `POSITIVE|NEUTRAL|NEGATIVE` as text). Projected gaze is joined later.

Label mapping used when reading folders of files:

```python
{'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
```

## Stage 1 — MATLAB to per-subject CSV

`read_ZuCo_mat.py` instantiates:

```python
DataTransformer('task1', level='sentence', scaling='raw', fillna='zeros')
```

and writes `1_SR.csv` … `12_SR.csv`. Committed copies live in `ZuCo_et_csv_data/`. Word-level exports live in `ZuCo_et_csv_data/word/`.

Raw (unscaled) sentence columns:

```text
id, SentLen, omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
```

Raw word columns:

```text
id, Sent_ID, Word_ID, Word, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT, WordLen
```

`Sent_ID` looks like `0_NR` for normal reading. The integer before the underscore is the sentence index used when joining text.

## Stage 2 — average across subjects

Readers differ in pace and skip rate. Downstream models use the **mean across the twelve subjects** for each sentence or word.

- Sentence averages: `get_average_sentence_level.py` (script still points at `et_csv_data/`; the outputs in-repo are `ZuCo_et_csv_data/average_data.csv` plus min-max and standard-scaled siblings).
- Word averages: `ZuCo_et_csv_data/word/get_average.py` → `word_averages.csv` / `word_averages_v2.csv`.

Zeros in numeric columns are treated as missing (`NaN`) before the sentence-level mean so unfixated words do not drag the average to zero. After averaging, two scalers are fit:

| File | Scaler | Use |
| --- | --- | --- |
| `min_max_scaled_average_data.csv` | `MinMaxScaler` | bounded `[0, 1]` features |
| `standard_scaled_average_data.csv` | `StandardScaler` | z-scores; this is what `model_ZuCo_SST.py` reads after the join |

`id` is preserved and used as the join key onto SST sentence ids.

**Subject 3 is short.** `ZuCo_et_csv_data/3_SR.csv` has 299 rows with `id` 0–298. `DataTransformer` skips a block of Task-1 sentences for that reader, and `read_ZuCo_mat.py` then `reset_index`, so those 299 rows are reindexed rather than keeping the original ZuCo sentence numbers. The committed average table still has 400 rows: `get_average_sentence_level.py` means by DataFrame index after `read_csv`, so subject 3 only contributes to sentences 0–298. `examples/inspect_datasets.py` treats this file as expected and fails if any *other* subject file is short.

That reindex-then-mean behavior is a reason to regenerate averages by `id` if you re-export from MATLAB.

## Stage 3 — join text and gaze (ZuCo SST)

`ZuCo_SST_data/ssts_ZuCo.csv` has `sentence_id`, `sentence`, `sentiment_label`. Joining on `sentence_id == id` produces:

- `combined_sst_et_standard.csv` — z-scored gaze (training default)
- `combined_sst_et_min_max.csv` — min-max gaze

`ZuCo_SST_data/spilt.py` then cuts an 80 / 10 / 10 train / valid / test split (`train.csv`, `valid.csv`, `test.csv`). The **model script does not use that split**. It re-reads the combined file and runs stratified 5-fold CV, so the committed split is for inspection and for the CPU example baselines.

Class balance on the combined ZuCo file is small-n and slightly uneven. Always stratify; a random 80/10/10 can hide a missing class in the test slice.

## Stage 4 — project gaze onto full SST

Full SST has many more sentences than ZuCo recorded. The personal pipeline fills those rows with **predicted or transferred** word-level gaze, then aggregates to sentence level.

Relevant artifacts:

| Path | Role |
| --- | --- |
| `gaze_prediction/data/convert_zuco_data.py` | Rescale word averages into a compact `nFix, FFD, GPT, TRT, GD` schema |
| `gaze_prediction/data/prediction_test.csv` | Predicted word-level gaze (v1) |
| `gaze_prediction/data/prediction_test_v2.csv` | Predicted word-level gaze (v2), ~192k rows |
| `gaze_prediction/data/provo.csv` | PROVO-style word table used as an extra reference corpus |
| `SST_data/convert_sst_to_et.py` | Tokenize SST sentences and emit a zero-filled word-level gaze skeleton |
| `SST_data/combined_full_sst_et.csv` | Full SST sentences + five sentence-level gaze columns |
| `SST_data/spilt.py` | 80 / 10 / 10 → `train_full_sst.csv`, `valid_full_sst.csv`, `test_full_sst.csv` |

`convert_sst_to_et.py` keeps only alphabetic tokens (`^[A-Za-z]+$`) via NLTK `word_tokenize`. That is a stricter tokenization than the RoBERTa tokenizer used at train time. Word-level prediction and sentence-level training therefore do not share a tokenizer; they only share a sentence id.

The five full-SST gaze columns (`nFix, GD, TRT, FFD, GPT`) are already scaled in the committed CSVs. Do not scale them again inside `model_full_SST.py`.

## Stage 5 — what the training scripts consume

```text
ZuCo track  →  ZuCo_SST_data/combined_sst_et_standard.csv
Full track  →  SST_data/{train,valid,test}_full_sst.csv
```

Both expect a `sentence` string and a numeric `sentiment_label`. Gaze column names must match the list hardcoded in each script.

## Practical notes for later personal runs

1. **Do not re-fit scalers on a split.** The committed standard-scaled ZuCo table was fit on all sentences. If you regenerate features, fit the scaler on the training fold only or accept that the current numbers leak the test fold into the scale.
2. **Join keys are integers.** `ssts_ZuCo.csv` sorts by `sentence_id`. If you rebuild from `.txt` files, keep that sort or the average gaze rows will attach to the wrong reviews.
3. **Missing MATLAB files.** `get_matfiles()` asserts that each task folder has exactly twelve files. A partial download fails closed.
4. **Encoding.** All text IO in the conversion scripts uses `utf-8`.
5. **PROVO vs ZuCo.** `gaze_prediction/data/provo.csv` uses `fixProp` instead of `GD`. Do not concatenate it onto a ZuCo word table without renaming.

`examples/inspect_datasets.py` prints row counts, label histograms, and null counts for every committed table so you can confirm a regenerate step did not drop a class or an id.
