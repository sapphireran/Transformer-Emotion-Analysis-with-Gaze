# Datasets

All counts below exclude the header row. Paths are relative to the
repository root.

## Label convention

Every joined table uses integer sentiment labels:

| Integer | Meaning | Source folder / SST tag |
|---|---|---|
| `0` | negative | `NEGATIVE` |
| `1` | neutral | `NEUTRAL` |
| `2` | positive | `POSITIVE` |

`SST_data/stts_all_sentence_level.csv` still stores the string tags
(`POSITIVE` / `NEGATIVE` / `NEUTRAL`) in column 2. The join scripts map
those strings through the dictionary in `convert_full_SST.py` and
`ZuCo_SST_data/save_SST_data.py`.

---

## 1. Full Stanford Sentiment Treebank (`SST_data/`)

Movie-review sentences with polarity. Eye-tracking columns on this split
are **not** recorded in a tracker — they are projected features sitting
on the same five-name schema the ZuCo model uses (`nFix`, `GD`, `TRT`,
`FFD`, `GPT`). Values routinely go negative, which is only possible after
standardization (or a similarly centered transform). Treat them as
**predicted / transferred gaze**, not milliseconds.

### `stts_all_sentence_level.csv`

| Column | Type | Notes |
|---|---|---|
| *(unnamed)* sentence | string | first column, often quoted |
| polarity | string | `POSITIVE` / `NEGATIVE` / `NEUTRAL` |

11,853 sentences. This is the text source for `convert_sst_to_et.py`.

### `combined_full_sst_et.csv`

```
sentence_id, sentence, sentiment_label, nFix, GD, TRT, FFD, GPT
```

11,853 rows. `sentence_id` is an integer key that is **not** a dense
0…N-1 index after the train/valid/test split (the split shuffles).

### Fixed 80 / 10 / 10 split

Produced by `SST_data/spilt.py` (`train_test_split`, `test_size=0.2`,
then 50/50 on the hold-out, `random_state=42`):

| File | Rows | `0` neg | `1` neu | `2` pos |
|---|---:|---:|---:|---:|
| `train_full_sst.csv` | 9,482 | 3,710 | 1,833 | 3,939 |
| `valid_full_sst.csv` | 1,185 | 476 | 209 | 500 |
| `test_full_sst.csv` | 1,186 | 463 | 199 | 524 |

Neutral is the minority class (~19%) in every split. `model_full_SST.py`
reads these three files directly.

### `sst_et_test.csv`

Word-level **placeholder** table written by `convert_sst_to_et.py`:

```
sentence_id, word_id, word, nFix, FFD, GPT, TRT, GD
```

191,971 tokens. Alphabetic tokens only (`^[A-Za-z]+$`); other tokens are
dropped, and an empty sentence becomes a single `unknown` token. All five
ET columns are `0` in this file — it is a schema template, not
measurements. The filled-in counterpart is
`gaze_prediction/data/prediction_test_v2.csv` (same shape, non-zero
values).

---

## 2. ZuCo ∩ SST (`ZuCo_SST_data/`)

ZuCo Task 1 is natural reading of movie reviews. This folder keeps the
**400 sentences** that were aligned with SST polarity labels.

### `ssts_ZuCo.csv`

```
sentence_id, sentence, sentiment_label
```

400 rows. `sentence_id` is 0…399 in file order (sorted numerically when
built from the `all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt` dump).

Label counts: **123 / 137 / 140** (neg / neu / pos). Much closer to
balanced than full SST.

### Joined sentence-level ET

`combined_sst_et_standard.csv` and `combined_sst_et_min_max.csv` share
the schema:

```
sentence_id, sentence, sentiment_label,
omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
```

400 rows. The ET block is the **cross-subject average** of the twelve
`ZuCo_et_csv_data/{k}_SR.csv` files, then scaled:

| File | Scaling |
|---|---|
| `combined_sst_et_standard.csv` | z-score (`StandardScaler`) |
| `combined_sst_et_min_max.csv` | min-max to `[0, 1]` |

`model_ZuCo_SST.py` reads the **standard** join and keeps only
`nFixations, FFD, GPT, TRT, GD` — five of the eight numeric ET columns.

### Optional 80 / 10 / 10 split

`ZuCo_SST_data/spilt.py` writes `train.csv` (320), `valid.csv` (40),
`test.csv` (40). The training script does **not** use these files; it
runs `StratifiedKFold` on the full 400-row standard join instead. The
split is useful for ad-hoc checks and is validated by
`examples/data_integrity.py`.

---

## 3. Per-subject ZuCo sentence ET (`ZuCo_et_csv_data/`)

One CSV per reader, named `{1–12}_SR.csv` (subject index `0–11` plus
one). Schema:

```
id, SentLen, omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
```

`id` is the sentence index after `DataTransformer` drops known-bad
ZuCo trials.

| File | Data rows | Notes |
|---|---:|---|
| `1_SR.csv`, `2_SR.csv`, `4_SR.csv`–`12_SR.csv` | 400 | complete Task 1 set |
| `3_SR.csv` | **299** | subject index 2; `utils_ZuCo.py` skips sentences 150–249 and 399 |

`average_data.csv` is the row-wise mean across subjects (400 rows).
`min_max_scaled_average_data.csv` and `standard_scaled_average_data.csv`
are that average after sklearn scaling. Zeros in the per-subject files
are treated as missing (`NaN`) before averaging in
`get_average_sentence_level.py`.

### Word-level sibling (`ZuCo_et_csv_data/word/`)

```
id, Sent_ID, Word_ID, Word, nFixations, meanPupilSize,
GD, TRT, FFD, SFD, GPT, WordLen
```

`Sent_ID` looks like `12_NR` (sentence index + task tag; Task 1/2 use
`_NR`, Task 3 would use `_TSR`). Typical files have **7,129** word rows;
`3_SR.csv` has **5,293**.

`word_averages.csv` / `word_averages_v2.csv` are cross-subject means of
the seven ET columns, with `Word` / `WordLen` taken from subject 1.
`get_average.py` fills null words with `unknown` and null measures with
`0`.

---

## 4. Gaze-prediction tables (`gaze_prediction/data/`)

These files use the **predictor schema** (scaled roughly onto a 0–100
range by `convert_zuco_data.py`):

```
sentence_id, word_id, word, nFix, FFD, GPT, TRT, GD
```

| File | Rows | Notes |
|---|---:|---|
| `prediction_test.csv` | 1,751 | starts at `sentence_id=300` (a hold-out slice) |
| `prediction_test_v2.csv` | 191,971 | same cardinality as `SST_data/sst_et_test.csv` |
| `provo.csv` | 2,659 | PROVO norms; last column is `fixProp`, **not** `GD` |

`convert_zuco_data.py` min-max scales `nFixations` on its own range and
scales `FFD/GPT/TRT/GD` on a **shared** range, then multiplies by 100.

---

## 5. What is *not* in the checkout

- Raw ZuCo `.mat` files (`ZuCo_mat_data/task1/`, 12 files). Required
  only if you re-run `read_ZuCo_mat.py`.
- The `ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt` dump.
  `convert_full_SST.py` and `save_SST_data.py` expect it.
- Trained `.pth` weights. `model_full_SST.py` writes
  `models/best_{model_type}_model.pth` and will fail if `models/` does
  not exist.

---

## Integrity checks

`examples/data_integrity.py` asserts, among other things:

- the three full-SST splits are a partition of `combined_full_sst_et.csv`
- the ZuCo 80/10/10 split is a partition of
  `combined_sst_et_standard.csv`
- `ssts_ZuCo.csv` sentence ids line up with the joined ET tables
- subject 3 is the only short sentence-level file
- there are no NaNs in the files the training scripts read
