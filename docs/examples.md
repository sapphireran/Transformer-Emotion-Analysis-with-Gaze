# Examples

All scripts assume the repo root is on `PYTHONPATH` and that `requirements.txt` is installed. They only read committed CSVs.

```bash
PYTHONPATH=. python3 examples/01_inspect_datasets.py
PYTHONPATH=. python3 examples/02_gaze_feature_report.py
PYTHONPATH=. python3 examples/03_text_vs_gaze_baselines.py
PYTHONPATH=. python3 examples/04_sentence_walkthrough.py
PYTHONPATH=. python3 examples/05_full_sst_sample.py
# or:
PYTHONPATH=. python3 examples/run_all.py
```

Output directory: `examples/output/` (created on the first run, gitignored).

## 01 — Inspect datasets

Walks `tea_gaze.paths.DATASETS`, writes `dataset_inventory.md` and a small HTML page, and prints the 400-row ZuCo+SST label mix.

Use this when you add a CSV: if it is not in `DATASETS`, the inventory will not mention it.

## 02 — Gaze feature report

Loads raw sentence averages and the z-scored labeled join.

- Descriptive stats for the five fusion features (raw ms / counts)
- Pearson correlation matrix
- Mean z-scored feature by sentiment class
- Pairwise reader agreement on `nFixations` (handles the short subject-3 file)
- Histogram + heatmap PNG

This is the fastest way to see that TRT and nFixations move together, and that GPT is the heavy tail.

## 03 — Text vs gaze vs fusion

Stratified 5-fold CV, same seed as `model_ZuCo_SST.py`. Three logistic models:

- text (TF-IDF)
- gaze (5 columns)
- fusion (`hstack`)

Writes markdown, HTML, and JSON. Also dumps a gaze-only coefficient table fit on all 400 rows so you can see which class a long GPT pushes toward.

Expect the text model to be strong: these are movie-review sentences with obvious lexicon. Gaze-only should be weaker. Fusion is the interesting column — if it does not beat text, the 16-d transformer head may not either.

## 04 — Sentence walkthrough

Picks a short sentence per label, shows:

- gold vs predicted label from a fusion logistic fit on an 80% split
- the five z-scored sentence features
- the example tokenizer
- word-level raw averages from `word_averages_v2.csv`

This is the “what does one row actually contain?” page.

## 05 — Full SST sample

Contrasts split sizes, label priors, and feature scales between measured ZuCo+SST and predicted full SST. Random preview rows come from `train_full_sst.csv`.

Use this before you point `model_full_SST.py` at those files so you remember `nFix` is predicted.

## Tests that lock the examples to the real files

```bash
PYTHONPATH=. pytest -q
```

`tests/test_io_and_schema.py` asserts the 400-row label counts, the `nFix` alias on full SST, and that reader 3 is 299 rows. If you regenerate CSVs and those numbers change, the tests should fail on purpose.
