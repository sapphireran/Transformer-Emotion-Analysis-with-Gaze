# Reproduction

This page is the “what do I actually run” companion to the architecture
and pipeline notes. Commands assume the repository root as cwd.

## 0. Environment

```bash
# examples only (pandas / numpy / scikit-learn)
python -m pip install -r requirements.txt

# original training scripts (torch + transformers + datasets + …)
python -m pip install -r requirements-train.txt
```

GPU is optional for the examples, recommended for training. The scripts
pick `cuda` when `torch.cuda.is_available()` else `cpu`.

Create the checkpoint directory before a full-SST run:

```bash
mkdir -p models
```

## 1. Documentation examples (no weights, no GPU)

These read the committed CSVs and print reports to stdout. They exit
non-zero if an integrity check fails.

```bash
python examples/inspect_datasets.py
python examples/data_integrity.py
python examples/gaze_feature_report.py
python examples/gaze_only_baseline.py
python examples/late_fusion_demo.py
python examples/word_level_preview.py
python examples/sentence_walkthrough.py

python examples/run_all.py          # runs the seven scripts above
```

`run_all.py` also writes a copy of the combined stdout under
`examples/output/run_all.txt` when that directory is writable.

Expected runtime on a laptop-class CPU: well under a minute, except
`late_fusion_demo.py` (a few seconds of numpy SGD) and
`gaze_only_baseline.py` (five sklearn folds, usually a couple of
seconds).

## 2. Re-running the ZuCo 5-fold experiment

```bash
# optional: change model_type near the top of the file
python model_ZuCo_SST.py
```

Reads `ZuCo_SST_data/combined_sst_et_standard.csv`. First launch
downloads `roberta-base` (or `bert-base-uncased`) from Hugging Face.

What you should see:

- `Using device: …` and `Model type: …`
- 5 folds × 20 epochs of a tqdm bar
- a per-fold `Validation Acc / P / R / F1` line
- four `Average Validation …` lines at the end

There is no saved `.pth`. If you want a checkpoint, add
`torch.save` yourself or run the full-SST script instead.

To compare text-only vs text+gaze, edit `model_type` and run again.
Keep the same seed (`StratifiedKFold(..., random_state=42)`) so folds
match.

## 3. Re-running the full SST experiment

```bash
mkdir -p models
python model_full_SST.py
```

Reads:

- `SST_data/train_full_sst.csv`
- `SST_data/valid_full_sst.csv`
- `SST_data/test_full_sst.csv`

Writes `models/best_{model_type}_model.pth` whenever validation
accuracy improves. Reloads that file for the test pass.

**Read the test numbers with care.** The test loop currently replaces
`all_preds` / `all_labels` on every batch instead of appending, so the
printed test metrics are computed on the **last batch only**. Validation
metrics during training are fine (they `extend`). Patch sketched in
[known-quirks.md](known-quirks.md).

## 4. Regenerating CSVs from scratch

Only needed if you have the raw sources. None of these writers use the
same directory names as the committed tree, so pass paths or edit the
scripts first.

| Goal | Script | Input it expects | Output it writes |
|---|---|---|---|
| Per-subject sentence ET | `read_ZuCo_mat.py` | `ZuCo_mat_data/task1/*.mat` (via `utils_ZuCo.get_matfiles`) | `et_csv_data/{1-12}_SR.csv` |
| Average + scale sentence ET | `get_average_sentence_level.py` | `et_csv_data/{1-12}_SR.csv` | `et_csv_data/{min_max,standard}_scaled_average_data.csv` |
| Word-level subject average | `ZuCo_et_csv_data/word/get_average.py` | `{1-12}_SR.csv` in that folder | `word_averages_v2.csv` |
| ZuCo text + labels | `convert_full_SST.py` | `ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt` | `ZuCo_SST_data/ssts_ZuCo.csv` |
| Same, folder-local | `ZuCo_SST_data/save_SST_data.py` | `all/…/*.txt` | `output.csv` |
| SST 80/10/10 | `SST_data/spilt.py` | `combined_full_sst_et.csv` | `train/valid/test_full_sst.csv` |
| ZuCo 80/10/10 | `ZuCo_SST_data/spilt.py` | `combined_sst_et_standard.csv` | `train.csv`, `valid.csv`, `test.csv` |
| Word-level SST template | `SST_data/convert_sst_to_et.py` | `stts_all_sentence_level.csv` | `sst_et_test.csv` (zeros) |
| Predictor-schema convert | `gaze_prediction/data/convert_zuco_data.py` | `training_data/word_averages_v2.csv` | `sst_et_train_and_vaild_v2.csv` |

`nltk.download('punkt')` is required for `convert_sst_to_et.py`.

## 5. Hardware and time (order of magnitude)

These are planning numbers, not measurements from this checkout.

| Run | Rough cost |
|---|---|
| examples | CPU, seconds |
| `model_ZuCo_SST.py` (`roberta_eye_tracking`, 5×20 epochs, 400 rows, batch 16) | one GPU, minutes |
| `model_full_SST.py` (`roberta_eye_tracking`, 5 epochs, 9.5k rows, batch 256) | one GPU, tens of minutes; CPU is possible but slow |
| first HF download | ~500 MB for `roberta-base` |

`hidden_layer_size = 16` does not move the needle next to the 125M-param
encoder. Almost all time is the transformer forward/backward.

## 6. What “done” looks like for a comparison

A fair text-only vs text+gaze comparison on ZuCo:

1. Run `model_ZuCo_SST.py` with `model_type = 'roberta'`.
2. Run again with `model_type = 'roberta_eye_tracking'`.
3. Compare the **average** validation F1 (and accuracy) across the same
   five folds.
4. Optionally repeat for `bert` / `bert_eye_tracking`.

On full SST, compare **validation** metrics logged each epoch (those are
correct). Do not treat the printed test line as the official score until
the last-batch bug is fixed.

`examples/gaze_only_baseline.py` is the cheap sanity check *before* you
spend a GPU hour: if a logistic regressor on the 5 ET columns cannot
beat the majority-class dummy, late fusion is unlikely to help much
unless the transformer uses the ET vector in a non-linear way the linear
probe cannot see.
