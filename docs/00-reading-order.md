# Reading order and conventions

This folder is personal commentary for
`Transformer-Emotion-Analysis-with-Gaze`. It is written for a future version
of myself who has forgotten which CSV is z-scored, why there are two split
scripts spelled `spilt.py`, and which five gaze columns actually enter the
network.

Nothing here is generated from a paper template. Numbers were counted from
the CSVs in this clone (September 2026) using the stdlib scripts under
`examples/`. If you regenerate a file, re-run `python examples/inspect_datasets.py`
and treat *that* printout as source of truth.

## Suggested path through the notes

1. **This page** — names, label integers, path conventions.
2. **`01-research-notes.md`** — what I was trying to show, and what I was
   not trying to show.
3. **`02-gaze-feature-glossary.md`** — FFD vs GD vs TRT vs GPT, with the
   actual millisecond ranges from `average_data.csv`.
4. **`03-data-pipeline.md`** — MATLAB → per-subject CSV → 12-reader mean →
   scaled table → training script. Also the full-SST / predicted-gaze fork.
5. **`04-model-architecture.md`** — the 768 + 16 concat, and why RoBERTa
   still uses `pooler_output` in this code.
6. **`05-training-and-evaluation.md`** — epochs, batch sizes, CV vs hold-out,
   and the test-loop bug in `model_full_SST.py`.
7. **`06-worked-examples.md`** — sentence 0 walked at word level; a high-GPT
   negative review; a three-word positive.
8. **`07-personal-lab-notes.md`** — path mismatches, unused columns, things
   I would change on a second pass.

Then run the three example scripts. They do not train a model.

## Names I keep mixing up

| Name in code | What I mean |
| --- | --- |
| ZuCo task 1 / `SR` / `NR` | Sentiment Reading / Normal Reading. The 400 movie-review sentences. |
| `nFix` vs `nFixations` | Same quantity. Full SST uses the short name; ZuCo uses the long one. |
| `spilt.py` | Typo for `split.py`. There is one in each data folder. |
| `et_csv_data` | Path hard-coded in `read_ZuCo_mat.py` and `get_average_sentence_level.py`. The checked-in folder is `ZuCo_et_csv_data/`. |
| `hidden_layer_size = 16` | Width of the gaze tower, not a BERT hidden size. |
| `best_val_acc` | Variable name in `model_full_SST.py`. The nearby comment still says "F1". It is accuracy. |

## Label integers

```
0 = NEGATIVE
1 = NEUTRAL
2 = POSITIVE
```

`CrossEntropyLoss` in both training scripts is called as
`logits.view(-1, 3)`, so the class count is baked in. If you ever go binary,
change that `3` and `num_labels` together.

## Scaling conventions

- **raw** — milliseconds and counts, as in `ZuCo_et_csv_data/average_data.csv`
  and the per-subject `*_SR.csv` files.
- **min-max** — each feature independently mapped to `[0, 1]` across the 400
  sentences (`min_max_scaled_average_data.csv`, `combined_sst_et_min_max.csv`).
- **standard / z-score** — mean 0, std 1 across the 400 sentences
  (`standard_scaled_average_data.csv`, `combined_sst_et_standard.csv`).
  **This is what `model_ZuCo_SST.py` reads.**
- **full SST gaze columns** — already standardized in
  `SST_data/combined_full_sst_et.csv` (mean ≈ 0, std ≈ 1 on the 11,853 rows).
  Those values are *predicted* or transferred, not recorded by an eye tracker
  on SST readers.

Word-level files (`ZuCo_et_csv_data/word/*.csv`) stay in raw-ish units in
`word_averages_v2.csv`. Do not mix a z-scored sentence vector with a raw
word-level vector in the same plot without saying so.

## Subject index

ZuCo has 12 readers. Scripts index them `0..11`. Output files are named
`1_SR.csv` … `12_SR.csv` (`i+1`). `utils_ZuCo.py` also has special-case
skips for known missing blocks (subject 2 on task 1, subjects 6 and 11 on
task 2, subjects 3 / 7 / 11 on task 3). This clone only ships task-1
sentiment-reading CSVs.

## What is *not* in this clone

- The original ZuCo `.mat` files (`ZuCo_mat_data/`).
- Trained `models/best_*.pth` checkpoints.
- The `ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt` folders that
  `convert_full_SST.py` expects.
- A recorded GPU training log. The comments in the scripts are the only
  original documentation.

The derived tables are here, so commentary and examples can stay honest
without the MATLAB dump.
