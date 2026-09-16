# Reproduction

Personal checklist for regenerating tables and rerunning experiments. The checked-in CSVs already let you skip every step that needs ZuCo `.mat` files or a GPU.

## Environment

```bash
# inspection + example baselines (this is what CI-like pytest uses)
python3 -m pip install -r requirements.txt

# original transformer scripts
python3 -m pip install -r requirements-train.txt
```

Recommended: a venv. The training extras pull `torch` and `transformers`.

ZuCo Matlab files are **not** in git. If you need to rebuild the per-subject CSVs, place them at `ZuCo_mat_data/task1/` (and fix the `\\` separators in `get_matfiles` on Linux).

## Path A — do nothing, inspect what is here

```bash
python3 -m pytest
python3 examples/scripts/01_explore_zuco_sst.py
python3 examples/scripts/02_gaze_feature_report.py
python3 examples/scripts/03_gaze_only_baseline.py
python3 examples/scripts/04_split_sanity_check.py
python3 examples/scripts/05_word_level_scanpath.py
python3 examples/scripts/06_fusion_shape_check.py
```

These read only committed CSVs.

## Path B — rebuild ZuCo sentence averages

Requires the 12 Task 1 `.mat` files.

```bash
# 1. export per subject (writes et_csv_data/ — copy or symlink to ZuCo_et_csv_data/)
python3 read_ZuCo_mat.py

# 2. reader-mean + sklearn scalers
#    edit folder_path in get_average_sentence_level.py if your export dir differs
python3 get_average_sentence_level.py
```

Then join `ssts_ZuCo.csv` to the scaled average on `id` / `sentence_id`. There is no dedicated join script in the repo; the combined files were produced offline. `examples/gazekit/io.py` can load both sides if you want to write a new join.

## Path C — rebuild word averages

```bash
cd ZuCo_et_csv_data/word
python3 get_average.py
```

Output: `word_averages_v2.csv` in that folder. Optional rescale for foreign plotters:

```bash
# convert_zuco_data.py expects training_data/word_averages_v2.csv
python3 gaze_prediction/data/convert_zuco_data.py
```

## Path D — rebuild SST text tables

Needs the unlabeled/labeled `.txt` tree (`ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/`).

```bash
python3 convert_full_SST.py          # -> ZuCo_SST_data/ssts_ZuCo.csv
# or, from inside ZuCo_SST_data/:
python3 save_SST_data.py             # -> output.csv (same shape, relative paths)
```

Splits:

```bash
python3 ZuCo_SST_data/spilt.py       # train/valid/test on the 400
python3 SST_data/spilt.py            # train/valid/test on full SST
```

`SST_data/convert_sst_to_et.py` needs NLTK `punkt`:

```bash
python3 SST_data/convert_sst_to_et.py
```

## Path E — train Track A (ZuCo, 5-fold)

```bash
# edit model_type at the top of the file if you want a text-only ablation
python3 model_ZuCo_SST.py
```

Expect five validation lines and four "Average Validation …" lines. There is no checkpoint.

## Path F — train Track B (full SST)

```bash
mkdir -p models
python3 model_full_SST.py
```

Writes `models/best_{model_type}_model.pth`. Until the test-loop `extend` bug is fixed, do not quote the printed **test** line; quote the last **Validation** line or patch the script (see [limitations.md](limitations.md)).

## Suggested ablation grid

Run these as four values of `model_type` on the same machine/seed if you start adding seeds:

1. `roberta`
2. `roberta_eye_tracking`
3. `bert`
4. `bert_eye_tracking`

Keep `hidden_layer_size=16` first. If fusion loses, try a 2-layer gaze MLP with ReLU before concluding the channels are useless.

## Recording a run

Write a short note (personal, not committed) with:

- git SHA
- `model_type`
- GPU name
- whether the scaler was fit inside the fold
- the five fold metrics or the per-epoch val acc
- majority-class accuracy on the same split (`examples/scripts/03_gaze_only_baseline.py`)

That is enough to compare later personal runs without pretending this is a paper repo.
