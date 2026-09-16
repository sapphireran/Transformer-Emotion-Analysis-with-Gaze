# Reproduction

What a fresh clone of this personal repo can actually do.

## Always works offline from git

```bash
python -m pip install -r requirements-examples.txt
python examples/01_inspect_catalog.py
python examples/02_sentence_gaze_stats.py
python examples/03_label_balance.py
python examples/04_compare_scalers.py
python examples/05_word_level_profiles.py --sentence-id 0_NR
python examples/06_text_vs_gaze_baselines.py
python examples/07_split_audit.py
python examples/08_write_dataset_report.py
python -m pytest
```

These commands only read the checked-in CSVs. They do not need ZuCo MATLAB
files, SST `all/` folders, NLTK tokenizers, torch, or a GPU.

Outputs land in `examples/output/` (plots, baseline CSV) and
`docs/generated/dataset-report.md`.

## Works if you install the training stack

```bash
python -m pip install torch transformers datasets scikit-learn pandas tqdm
mkdir -p models
python model_ZuCo_SST.py
python model_full_SST.py
```

First run downloads `roberta-base` (or `bert-base-uncased`) from Hugging
Face. That needs network access. Five-fold × 20 epochs × a full encoder on
CPU is a long wait; use CUDA if you have it.

Edit the `model_type` assignment at the top of each script to run the
text-only controls.

Treat `model_full_SST.py` test numbers as last-batch-only until the
`.extend` bug is fixed.

## Does not work from this clone alone

| Step | Why it fails here |
| --- | --- |
| `read_ZuCo_mat.py` | `ZuCo_mat_data/` is not checked in; path is Windows-style |
| `convert_full_SST.py` / `ZuCo_SST_data/save_SST_data.py` | `all/{NEGATIVE,POSITIVE,NEUTRAL}/` is not checked in |
| `SST_data/convert_sst_to_et.py` | needs NLTK `punkt` and is meant to be run inside `SST_data/` |
| `get_average_sentence_level.py` | looks for `et_csv_data/`, not `ZuCo_et_csv_data/` |
| Gaze-predictor training | no training script, only Provo + prediction CSVs |

The derived products of those scripts **are** in git, so analysis and
transformer training do not depend on them.

## Expected file checks

`examples/07_split_audit.py` exits non-zero if either stored split family
leaks ids or fails to cover its parent. `tests/test_splits.py` asserts the
same plus the exact ZuCo 320/40/40 sizes.

`tests/test_scaling.py` asserts that `combined_sst_et_standard.csv` is
standard-like and `combined_sst_et_min_max.csv` is min-max-like. If a future
export forgets to scale, CI-style pytest will fail.

## Seeds

| Location | Seed |
| --- | --- |
| ZuCo `StratifiedKFold` | 42 |
| `spilt.py` train/valid/test | 42 |
| sklearn example baselines | 42 (override with care; folds must stay aligned) |
| `GazeFusionForward` demo weights | 0 |

PyTorch training itself does not call `torch.manual_seed` or set
`cudnn.deterministic`. Transformer runs will move between machines.

## Environment notes

`requirements-examples.txt` pins only lower bounds. The examples were
developed against Python 3.12, pandas 2, numpy 1.26+, scikit-learn 1.3+.
Training scripts were written against an older `transformers` API
(`from_pretrained` on `BertModel` / `RobertaModel`) that is still current.
