# Examples

Lightweight, **CPU-only** scripts for this personal repo. They read the
committed CSVs and write text under `sample_outputs/`. They do **not**
download BERT/RoBERTa, do not need `pandas` / `sklearn` / `torch`, and
do not replace `model_ZuCo_SST.py` or `model_full_SST.py`.

Need: Python 3.10+ and `numpy` (already installed in most scientific
environments; `requirements.txt` lists it too).

Checked-in copies of the last local run live in `sample_outputs/`.
They are snapshots for reading on GitHub; re-run the scripts to
refresh them. Numbers are interpreted in
[docs/09-example-baselines.md](../docs/09-example-baselines.md).

Run from the repository root or from `examples/` — helpers resolve
paths from this file’s location.

```bash
python3 examples/01_inspect_datasets.py
python3 examples/02_schema_check.py
python3 examples/03_gaze_feature_summary.py
python3 examples/04_label_and_length.py
python3 examples/05_word_level_gaze.py
python3 examples/06_toy_late_fusion.py
python3 examples/07_subject_variance.py
python3 examples/08_split_integrity.py

# or
python3 examples/run_all.py
```

`02_schema_check.py` and `08_split_integrity.py` should stay green.
They exit `1` if a header, row count, label alphabet, or split
partition drifts.

## What each script is for

| Script | Reads | Writes | Question it answers |
| --- | --- | --- | --- |
| `01_inspect_datasets.py` | every major CSV | `01_inspect_datasets.txt` | What is in the tree, and is class balance what I think? |
| `02_schema_check.py` | the same tables | `02_schema_check.txt` | Did a regenerate silently break a contract? |
| `03_gaze_feature_summary.py` | ZuCo 400 + SST train | `03_gaze_feature_summary.txt` | Means, correlations, class-conditional gaze |
| `04_label_and_length.py` | ZuCo 400 + SST train | `04_label_and_length.txt` | Is “sentiment” just “sentence length”? |
| `05_word_level_gaze.py` | word averages v2 | `05_word_level_gaze.txt` | What does a single sentence look like token-by-token? |
| `06_toy_late_fusion.py` | ZuCo 400 | `06_toy_late_fusion.txt` | Does 5-d gaze beat majority in a linear model? |
| `07_subject_variance.py` | 12 `*_SR.csv` | `07_subject_variance.txt` | How much do readers disagree before averaging? |
| `08_split_integrity.py` | combined + splits | `08_split_integrity.txt` | Are the committed splits a clean partition? |

Shared parsing / metrics live in `common.py` so the eight scripts stay
short and the KFold + weighted F1 match the story in
`docs/06-training-loops.md`.

## What the toy fusion is not

`06_toy_late_fusion.py` trains a softmax on:

- 5 gaze numbers, or
- 7 hand-built text stats (counts + a tiny polarity list), or
- both concatenated

That is the **shape** of late fusion (concat, then a linear classifier),
not the **model** in `EyeTrackingModel`. A win here is permission to
fine-tune RoBERTa, not a paper table.

## Adding another example

1. Import helpers from `common.py`.
2. Use `read_csv("ZuCo_SST_data/...")` — never hard-code `/workspace`.
3. Print to stdout **and** `write_text("NN_name.txt", ...)`.
4. Mention the script in this README and in the root README.
5. Keep dependencies at stdlib + numpy.
