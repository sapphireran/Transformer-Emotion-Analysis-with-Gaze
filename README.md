# Transformer Emotion Analysis with Gaze

Personal research repo for **3-class sentiment classification** that concatenates a transformer sentence embedding with five eye-tracking features.

There is no company code here. The measured gaze comes from the public [ZuCo](https://osf.io/q3zws/) normal-reading task (movie-review sentences). Labels come from the Stanford Sentiment Treebank. The larger SST tables in `SST_data/` carry **predicted** gaze, not new recordings.

This checkout started as training scripts plus CSVs and no documentation. The `docs/` and `examples/` trees, and the small `tea_gaze` helper package, are the personal notes and runnable companions for those scripts.

## What is in the box

| Path | Role |
|---|---|
| `model_ZuCo_SST.py` | BERT / RoBERTa ± gaze, 5-fold CV on the 400 ZuCo+SST sentences |
| `model_full_SST.py` | Same fusion model on the large SST split with predicted gaze |
| `utils_ZuCo.py` | MATLAB → sentence/word tables (`DataTransformer`) |
| `ZuCo_et_csv_data/` | Per-reader and averaged eye-tracking extracts |
| `ZuCo_SST_data/` | 400 labeled sentences joined to those extracts |
| `SST_data/` | Full SST + predicted sentence-level gaze |
| `gaze_prediction/` | Word-level predicted gaze and a PROVO reference table |
| `tea_gaze/` | Schemas, feature glossary, CPU baselines used by the examples |
| `docs/` | Data, features, architecture, known issues |
| `examples/` | Inventory, feature report, text vs gaze vs fusion, walkthroughs |

Sentiment labels are `0 = NEGATIVE`, `1 = NEUTRAL`, `2 = POSITIVE`.

## Two experiment tracks

**Track A — measured gaze (small, real ET).**
`ZuCo_SST_data/combined_sst_et_standard.csv` has 400 sentences (123 / 137 / 140).
`model_ZuCo_SST.py` runs `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` for 20 epochs, batch size 16.

**Track B — predicted gaze (large SST).**
`SST_data/train_full_sst.csv` (9482), `valid_full_sst.csv` (1185), `test_full_sst.csv` (1186).
`model_full_SST.py` trains 5 epochs, batch size 256, and writes `models/best_{model_type}_model.pth`.

Both scripts share the same fusion head: transformer `pooler_output` plus a 16-unit projection of `(nFixations|nFix, FFD, GPT, TRT, GD)`.

## CPU examples (no GPU, no Hugging Face weights)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest -q
PYTHONPATH=. python3 examples/run_all.py
```

On the 400-sentence ZuCo+SST table the CPU baselines (5-fold, seed 42) scored **text F1 0.462**, **gaze-only F1 0.337**, **fusion F1 0.468**. Details: [docs/sample-results.md](docs/sample-results.md).

Reports land in `examples/output/` (gitignored). Start with [docs/overview.md](docs/overview.md) and [examples/README.md](examples/README.md).

## Transformer training (original scripts)

The scripts still hard-code `model_type` and local CSV paths. They need PyTorch, `transformers`, `datasets`, and a GPU if you want the published-style run to finish in a reasonable time.

```text
model_type = 'roberta_eye_tracking'   # or bert, roberta, bert_eye_tracking
```

See [docs/model-architecture.md](docs/model-architecture.md) and [docs/known-issues.md](docs/known-issues.md) before launching a long job. The full-SST test loop currently keeps only the last batch; that is documented rather than silently “fixed” in a drive-by edit.

## Datasets (credits)

- Hollenstein, N., et al. *ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading.* Scientific Data (2018).
- Socher, R., et al. *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.* EMNLP (2013).
- Luke, S. G., & Christianson, K. *The Provo Corpus: A large eye-tracking corpus with predictability ratings.* Behavior Research Methods (2018).

Use those sources’ licenses if you redistribute the raw corpora. The CSVs here are personal working extracts.
