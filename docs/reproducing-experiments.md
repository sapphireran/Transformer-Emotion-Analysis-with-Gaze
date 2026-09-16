# Reproducing experiments

Two layers of reproduction:

1. **CSV and math examples** — no GPU, no Hugging Face download. This is
   what `examples/` and `tests/` cover.
2. **Original trainers** — full BERT/RoBERTa fine-tunes. Expect a CUDA box
   and several gigabytes of cached weights.

## 1. Docs / examples environment

```bash
python3 -m pip install -r requirements-dev.txt
bash examples/run_all.sh
python3 -m pytest tests -q
```

Individual scripts:

```bash
python3 examples/inspect_datasets.py
python3 examples/gaze_feature_tour.py
python3 examples/label_and_split_audit.py
python3 examples/fusion_architecture_demo.py
python3 examples/synthetic_training_loop.py
```

Each script prints a self-contained report and exits 0 on success. They
read only files already in git.

## 2. ZuCo 5-fold fine-tune

Prereqs: the 400-row combined table (already present) plus

```bash
python3 -m pip install torch transformers datasets tqdm scikit-learn pandas
```

Then:

```bash
# optional: export CUDA_VISIBLE_DEVICES=0
python3 model_ZuCo_SST.py
```

You should see `Using device: cuda` (or `cpu`), `Model type: roberta_eye_tracking`,
20 progress bars per fold, a per-fold validation line, and four "Average
Validation …" summaries.

Wall time is dominated by downloading `roberta-base` once and then five
20-epoch fine-tunes on 320-ish sentences each. On CPU this is slow; use a
GPU or cut `num_epochs` for a smoke test.

To run text-only, edit the top of the file:

```python
model_type = 'roberta'
```

## 3. Full-SST hold-out fine-tune

```bash
mkdir -p models
python3 model_full_SST.py
```

Creates `models/best_roberta_eye_tracking_model.pth` whenever validation
accuracy improves. After 5 epochs it reloads that file and prints a test
line. **Do not trust the test line until the last-batch overwrite described
in known-issues.md is fixed.**

## 4. Regenerating ET tables from MATLAB

Only needed if you obtain the official ZuCo `.mat` release.

1. Place 12 Task-1 files under `ZuCo_mat_data/task1/` (and fix the
   backslash in `get_matfiles` on Linux).
2. Point `read_ZuCo_mat.py` at `ZuCo_et_csv_data` instead of `et_csv_data`.
3. Run `read_ZuCo_mat.py`, then `get_average_sentence_level.py` with the
   same folder.
4. Confirm subject 3 still has ~299 sentence rows.

Word-level regeneration uses `DataTransformer(..., level='word')` and
`ZuCo_et_csv_data/word/get_average.py`.

## 5. Regenerating the 80/10/10 splits

```bash
cd ZuCo_SST_data && python3 spilt.py
cd ../SST_data && python3 spilt.py
```

Both scripts hard-code `random_state=42`. Re-running them should recreate
the same files if pandas/sklearn shuffle behavior matches.

## 6. Hardware notes

| Job | Memory | Disk | Notes |
| --- | --- | --- | --- |
| examples + pytest | < 500 MB | checked-in CSVs only | CPU |
| ZuCo 5-fold RoBERTa | ~6 GB GPU | HF cache ~500 MB | batch 16, length 128 |
| Full SST RoBERTa | ~8 GB GPU | same cache + `models/` | batch 256 |

`batch_size = 256` at length 128 is comfortable on a 12 GB card and tight
on 8 GB. Drop it if you OOM; metrics stay comparable if you keep Adam at
`5e-5`.

## 7. Sanity checks before quoting numbers

- [ ] Same `model_type` string in the log and the checkpoint name
- [ ] ZuCo numbers are **means over 5 folds**, not a single 40-row test file
- [ ] Full-SST test loop uses `extend`, not `=`
- [ ] Weighted F1 is reported alongside accuracy
- [ ] Text-only and fusion runs used the same tokenizer and max length
