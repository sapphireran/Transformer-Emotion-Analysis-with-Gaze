# Reproduction notes

This page is the “I have a fresh clone, what do I actually run?” list for a
personal checkout. It does not assume MATLAB ZuCo dumps are present — those
are not in git.

## 0. Environment

Python 3.10+ recommended. Examples:

```bash
python3 -m pip install -r requirements-examples.txt
python3 examples/run_all.py
python3 -m unittest discover -s examples/tests -v
```

Training:

```bash
python3 -m pip install -r requirements.txt
# optional, only if you rebuild sst_et_test.csv
python3 -c "import nltk; nltk.download('punkt')"
```

GPU is optional for the examples, recommended for the two `model_*.py` scripts.

## 1. Use the checked-in tables (usual path)

You do **not** need MATLAB to train:

```bash
mkdir -p models
python3 model_ZuCo_SST.py      # 400 rows, 5-fold CV
python3 model_full_SST.py      # 11,853 rows, hold-out
```

Edit `model_type` in-file for ablations. Paths inside the scripts are relative
to the **repository root**. Run them from there.

## 2. Rebuild ZuCo sentence CSVs (needs local MATLAB dumps)

1. Place 12 Task-1 `.mat` files under `ZuCo_mat_data/task1/`.
2. In `utils_ZuCo.get_matfiles`, change the default `subdir` from
   `'\\ZuCo_mat_data\\'` to `'ZuCo_mat_data'` (POSIX).
3. In `read_ZuCo_mat.py`, write to `ZuCo_et_csv_data` instead of `et_csv_data`.
4. `python3 read_ZuCo_mat.py`
5. Point `get_average_sentence_level.py` at `ZuCo_et_csv_data` and run it.
6. Join `standard_scaled_average_data.csv` to `ssts_ZuCo.csv` on `id` /
   `sentence_id` to refresh `combined_sst_et_standard.csv`.

Word-level: set `DataTransformer(..., level='word')` and reuse the skip logic.
`ZuCo_et_csv_data/word/get_average.py` expects to be run with that folder as
the working directory.

## 3. Rebuild ZuCo labels from text files

If you unzip `ZuCo_SST_data/ZuCo_SST_data.zip` and get
`all/NEGATIVE|POSITIVE|NEUTRAL/*.txt`:

```bash
python3 convert_full_SST.py
# or, from ZuCo_SST_data/:
python3 save_SST_data.py
```

`convert_full_SST.py` writes `ZuCo_SST_data/ssts_ZuCo.csv`.
`save_SST_data.py` writes `output.csv` in the current directory — rename it.

## 4. Rebuild the full-SST hold-out

```bash
cd SST_data
python3 spilt.py    # reads combined_full_sst_et.csv
```

That overwrites `train_full_sst.csv`, `valid_full_sst.csv`, `test_full_sst.csv`
with the same 80 / 10 / 10, `random_state=42` cut that is already committed.

To regenerate the zero-filled word skeleton:

```bash
cd SST_data
python3 convert_sst_to_et.py   # needs nltk punkt; input is headerless
```

The input file has **no header**. `csv.reader` treats the first row as data,
which is what you want, but any future `DictReader` will be wrong.

## 5. Sanity checks that should stay green

These do not need torch or network access:

| Command | What it proves |
| --- | --- |
| `python3 examples/inspect_datasets.py` | Schemas and row counts match [datasets.md](datasets.md) |
| `python3 examples/split_integrity.py` | Train/valid/test are a disjoint cover of combined |
| `python3 examples/scaling_check.py` | Scaled averages match numpy min-max / z-score |
| `python3 examples/join_check.py` | Combined ZuCo tables = labels ⨝ scaled averages |
| `python3 examples/word_to_sentence.py` | Subject-1 word means reconstruct sentence `nFixations` |
| `python3 examples/fusion_forward.py` | Fusion shapes are `(B, 3)` |
| `python3 examples/gaze_by_sentiment.py` | Class-conditional gaze means reprint |

## 6. Training output you can trust

**ZuCo script.** The four `Average Validation …` lines are means over 5 folds
of the *last* epoch. Save `val_accs` if you want a spread (min / max / std).

**Full-SST script.** Trust the **validation** line printed each epoch. Do not
trust the final **Test** line until the `all_preds = preds` bug is fixed
([known-issues.md](known-issues.md)). After a fix, the test line is a single
hold-out number, not a CV mean.

## 7. Seeds

| Location | Seed |
| --- | --- |
| ZuCo `StratifiedKFold` | 42 |
| Both `spilt.py` files | 42 |
| `torch` / `numpy` / `random` | **unset** |

Weight initialization and dropout therefore vary across training runs. The
**data splits** do not.

## 8. Artifacts that should not be committed

`.gitignore` already drops `models/`, `*.pth`, virtualenvs, and caches.
Keep predicted-gaze CSVs out of new commits unless they are small; 
`prediction_test_v2.csv` is already ~12 MB.
