# Data pipeline

This page is the intended order of operations. Several scripts still point at folders that are not in git (`ZuCo_mat_data/`, `et_csv_data/`, `ZuCo_SST_data/all/`). The **outputs** of those steps *are* checked in, so you can train without MATLAB. See [known-issues.md](known-issues.md) for the path mismatches.

## Stage 0 — Source corpora (not all checked in)

1. **ZuCo 1.0**, sentiment-reading task (task 1 / SR). Twelve subjects, MATLAB `sentenceData` structs. Official distribution is separate from this repo. `utils_ZuCo.get_matfiles()` expects `ZuCo_mat_data/task1/` with exactly 12 `.mat` files.
2. **Stanford Sentiment Treebank** sentence-level labels. The checked-in `SST_data/stts_all_sentence_level.csv` is an export with string labels and no header.
3. Optional: **Provo** word-level reading measures, sampled in `gaze_prediction/data/provo.csv`.

## Stage 1 — ZuCo MATLAB → per-subject CSV

```
read_ZuCo_mat.py
    DataTransformer('task1', level='sentence', scaling='raw', fillna='zeros')
    for subject in 0..11:
        write et_csv_data/{subject+1}_SR.csv
```

What `DataTransformer` does at sentence level:

1. Load `sentenceData` with `scipy.io.loadmat(..., squeeze_me=True, struct_as_record=False)`.
2. Skip known bad sentence ranges for specific subject/task pairs (task 1 / subject 2 drops indices 150–249 and 399).
3. For each remaining sentence, walk `sent.word`:
   - Pull `nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT` if the attribute exists and is not an `ndarray`; otherwise 0.
   - Accumulate those values and count how many words were actually fixated (`nwords_fixated`).
4. Set `SentLen = len(sent.word)` and `omissionRate = sent.omissionRate`.
5. Divide the accumulated duration/count features by `nwords_fixated` (mean over fixated words, not over all tokens).
6. Drop rows containing `±inf`.
7. Optionally scale (`raw` in `read_ZuCo_mat.py`, so this step is a no-op).
8. Fill remaining NaNs with zeros / min / mean.

Word-level mode writes one row per token with `Sent_ID` like `{idx}_NR` and `WordLen`. Those files live under `ZuCo_et_csv_data/word/`.

`split_data()` in `utils_ZuCo.py` exists to control order effects on task 1 (first half vs second half of each subject). It is not called by the training scripts.

## Stage 2 — Average subjects and scale

```
get_average_sentence_level.py
    read et_csv_data/{1-12}_SR.csv
    replace 0 with NaN on feature columns
    mean across subjects by row index
    write average_data.csv                 # (this file is already in ZuCo_et_csv_data/)
    MinMaxScaler  → min_max_scaled_average_data.csv
    StandardScaler → standard_scaled_average_data.csv
```

The checked-in averages are under `ZuCo_et_csv_data/`. The script still says `folder_path = 'et_csv_data'`. When you re-run it, point that path at `ZuCo_et_csv_data` or copy the subject files.

Word-level analogue: `ZuCo_et_csv_data/word/get_average.py` concatenates the twelve word CSVs, `groupby(level=0).mean()` on the numeric columns, and stitches `id, Sent_ID, Word_ID, Word, WordLen` from subject 1. Empty words become `unknown`; empty numerics become 0. Output: `word_averages_v2.csv`.

Because subject 3 has fewer rows, averaging **by row index** is not the same as averaging **by sentence id**. Treat `word_averages_v2.csv` as a convenience table, not a gold alignment. `examples/aggregate_subjects.py` re-aggregates sentence-level files **by `id`**. That matches the checked-in `average_data.csv`, but subject 3's `id` is compacted after the skipped sentences (see [known-issues.md](known-issues.md)). `examples/realign_subject3.py` maps those ids back to the original sentence index before averaging.

## Stage 3 — Attach sentiment labels (ZuCo)

```
ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt     # not checked in
        │
        ▼
convert_full_SST.py   or   ZuCo_SST_data/save_SST_data.py
        │
        ▼
ssts_ZuCo.csv          # 400 rows: sentence_id, sentence, sentiment_label
        │
        join on sentence_id == average id
        ▼
combined_sst_et_standard.csv
combined_sst_et_min_max.csv
```

Label mapping in both converters:

```
NEGATIVE → 0
NEUTRAL  → 1
POSITIVE → 2
```

`convert_full_SST.py` writes `ZuCo_SST_data/ssts_ZuCo.csv` and parses the filename stem as an integer id. `save_SST_data.py` writes `output.csv` and keeps the stem as a string. The checked-in `ssts_ZuCo.csv` matches the integer-id version.

The join itself is not a committed script; the joined CSVs are. `examples/split_sanity_check.py` confirms `sentence_id` sets match between `ssts_ZuCo.csv` and the combined tables.

Optional 80/10/10 from the standard combined table:

```
ZuCo_SST_data/spilt.py
    combined_sst_et_standard.csv
    → train.csv (320) / valid.csv (40) / test.csv (40)
```

## Stage 4 — Full SST + transferred gaze

```
SST_data/stts_all_sentence_level.csv          # sentence, string label; no header
        │
        ▼
SST_data/convert_sst_to_et.py
        NLTK word_tokenize, keep [A-Za-z]+
        → sst_et_test.csv                     # word rows, gaze = 0
        │
        ▼
external gaze predictor (not in this repo)
        → gaze_prediction/data/prediction_test_v2.csv
        │
        sentence-level reduce + standardize
        ▼
SST_data/combined_full_sst_et.csv
        │
        ▼
SST_data/spilt.py
        → train_full_sst.csv / valid_full_sst.csv / test_full_sst.csv
```

`gaze_prediction/data/convert_zuco_data.py` shows one scaling convention used on the **training** side of the predictor: min–max `nFixations` independently, min–max `{FFD,GPT,TRT,GD}` together, then multiply by 100. That is why predicted word-level nFix clusters around 20 and durations around 4–8 instead of millisecond-scale numbers.

`examples/reduce_predicted_gaze.py` mean-pools `prediction_test_v2.csv` by `sentence_id` and correlates with `combined_full_sst_et.csv`. On this checkout every SST sentence id is present and the token skeleton matches (`token equality = 1.0`), but Pearson r is only 0.58 / 0.41 / 0.46 / 0.64 / **0.06** for nFix / FFD / GPT / TRT / GD. The checked-in sentence vectors are **not** a plain mean of that word file — GD especially looks like a different reducer or a different source.

`result/*.png` are pairwise scatter/histograms of those predicted features (train, test, Provo). They are a distribution check, not a model score.

## Stage 5 — Train

```
model_ZuCo_SST.py
    combined_sst_et_standard.csv
    5-fold stratified CV

model_full_SST.py
    SST_data/{train,valid,test}_full_sst.csv
    holdout + best-accuracy checkpoint
```

Details: [models-and-training.md](models-and-training.md).

## Column-name bridge

When you write new glue code, normalize names early:

```
nFixations  ↔  nFix
FFD         ↔  FFD
GPT         ↔  GPT
TRT         ↔  TRT
GD          ↔  GD
```

`examples/common.py` exposes `ZUCO_FUSION_COLUMNS` and `SST_FUSION_COLUMNS` in the order the PyTorch datasets use: nFix, FFD, GPT, TRT, GD.

## What you can skip

If you only want to inspect or re-train on the checked-in tables:

- Skip Stage 0–1 (no MATLAB needed).
- Skip Stage 2–4 unless you are regenerating a CSV.
- Run Stage 5, or stay in `examples/` if you do not want to download `roberta-base`.
