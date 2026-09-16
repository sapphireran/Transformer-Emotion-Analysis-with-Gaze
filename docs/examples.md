# Examples guide

The `examples/` directory is a standalone, standard-library toolkit for inspecting the checked-in CSVs and for rehearsing the fusion arithmetic without downloading BERT or RoBERTa.

```bash
python3 examples/inspect_datasets.py
python3 examples/schema_check.py
python3 examples/label_balance.py
python3 examples/split_audit.py
python3 examples/gaze_feature_report.py
python3 examples/fusion_forward_demo.py
python3 examples/word_level_preview.py
python3 examples/generate_markdown_tables.py
```

Every script prints to stdout. Reports also land in `examples/outputs/` when you pass `--write` (all scripts accept it except where noted).

## Design rules

1. **No third-party imports.** If `pandas` / `torch` / `sklearn` are missing, the examples still run.
2. **Read-only on source CSVs.** Scripts never overwrite `SST_data/` or `ZuCo_*`.
3. **Repo-root independent cwd.** Paths resolve from the file location, so `python3 examples/inspect_datasets.py` works from any directory.
4. **Deterministic demos.** Splits and hashed embeddings use fixed seeds / `hashlib`, not `hash()`.

## Library map (`examples/lib/`)

| Module | Responsibility |
| --- | --- |
| `paths` | repo root and named CSV locations |
| `io_csv` | DictReader wrapper, typed numeric columns |
| `schema` | expected columns and alias maps |
| `stats` | mean, std, quantiles, histograms |
| `features` | named gaze sets, class-conditional means |
| `linalg` | small dense linear algebra (lists of lists) |
| `fusion` | hash embed, linear layers, SGD softmax |
| `metrics` | accuracy, macro/weighted P/R/F1, confusion |
| `report` | markdown / plain-text tables |

`tests/test_examples_lib.py` covers the numeric helpers with `unittest`.

## What each script is for

### `inspect_datasets.py`

Walks the named tables and prints row counts, columns, and a couple of example sentences. Use this when you clone the repo and want to confirm the data files opened.

### `schema_check.py`

Compares each table to the schema in `examples/lib/schema.py`. Fails (exit code 1) if a required column is missing or a gaze column is non-numeric. Extra columns are warnings.

### `label_balance.py`

Prints label histograms and majority-class accuracy for every sentiment table. This is the number a model must beat.

### `split_audit.py`

Checks that train + valid + test row counts add up, that `sentence_id` sets are disjoint, and that no split invented ids that are absent from the combined file.

### `gaze_feature_report.py`

For `zuco5`, `zuco_full`, and `sst5`, prints per-class means and a one-dimensional association score (absolute difference of class means, averaged). This is not a significance test. It is a "which columns even move?" ranking.

### `fusion_forward_demo.py`

Trains tiny softmax classifiers on ZuCo:

- majority class
- gaze-only (5 z-scored features)
- text-only (32-d hashed bag of words)
- concat-linear (hash ⊕ raw 5-d gaze)
- two-layer fused head (`Linear(gaze)` then concat then `Linear`), matching `docs/architecture.md`
- the same two-layer head with row-shuffled gaze (ablation)

Reports accuracy and macro F1 on a stratified holdout. This is the example to run when you want a number without a GPU.

### `word_level_preview.py`

Joins the 400 labeled ZuCo sentences to `word_averages_v2.csv` and prints per-token `nFixations` / `FFD` / `GD` / `TRT` / `GPT` for a few reviews, plus the words with the longest total reading time.

### `generate_markdown_tables.py`

Rebuilds `examples/outputs/dataset_inventory.md` from the live CSVs so the docs stay honest if a table is replaced.

## Expected runtime

All scripts except the fusion demo finish in a second or two. The demo trains for a few hundred SGD steps on 400 rows and should stay under ten seconds on CPU.

## Extending

Add a new named table in `examples/lib/paths.py` and a schema in `examples/lib/schema.py`. `inspect_datasets.py` and `schema_check.py` will pick it up automatically if you also register it in `TABLES`.
