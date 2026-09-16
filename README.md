# Transformer Emotion Analysis with Gaze

Personal research checkout: **gaze as a late-fusion sidecar** on SST movie
reviews. Two tracks share an `EyeTrackingModel` that concatenates a 16-d
linear projection of five eye-tracking scalars with a BERT/RoBERTa pooler
(`768 + 16 = 784`), then classifies three sentiment labels.

This branch adds a **Gaze Sidecar Atlas** — CPU-only docs and examples that
measure what is actually in the committed CSVs. They do not download
`roberta-base` and they do not claim to reproduce 2024 GPU scores.

```bash
python3 -m pip install -r requirements.txt
python3 examples/run_all.py
python3 -m unittest discover -s tests -v
```

Open `examples/outputs/sidecar_atlas.html` after `run_all.py` for the combined
lab view.

## The two tracks

| Track | Table | Gaze source | Trainer | Protocol |
| --- | --- | --- | --- | --- |
| ZuCo-SST | `ZuCo_SST_data/combined_sst_et_standard.csv` (400 rows) | Measured, then z-scored, from ZuCo Task 1 readers | `model_ZuCo_SST.py` | 5-fold stratified CV, 20 epochs, batch 16 |
| Full SST | `SST_data/{train,valid,test}_full_sst.csv` (~11.8k) | Predicted / imputed, already z-scored | `model_full_SST.py` | 80/10/10 split, 5 epochs, batch 256 |

Only **three** review strings appear in both corpora. Measured gaze and
predicted gaze are not two views of the same items. See
[`docs/two-tracks.md`](docs/two-tracks.md).

## What the atlas found in the tables

These numbers are recomputed by `examples/` (not remembered from a notebook):

- **Sidecar rank.** On full-SST train, PC1 explains ~91.7% of the five gaze
  columns; PC2 (almost entirely GD) another ~8.2%. A `Linear(5, 16)` sidecar
  is overcomplete. [`docs/sidecar-geometry.md`](docs/sidecar-geometry.md)
- **Subject 3 is packed, not just short.** Task-1 MATLAB subject 2 skips
  sentences 150–249 and 399, then writes ids `0..298`. CSV row 150 is
  original sentence 250. Row-wise `groupby(level=0).mean()` therefore mixes
  different reviews after the hole. [`docs/subject-3-reindex.md`](docs/subject-3-reindex.md)
- **Test metrics are last-batch only.** `model_full_SST.py` assigns
  `all_preds = preds.cpu().numpy()` instead of `extend`. With test n = 1186
  and batch 256, the printed test score is 162 rows (~13.7%).
  [`docs/known-issues.md`](docs/known-issues.md)
- **Gaze vs sentiment is weak.** |r| ≈ 0.06 on full SST, even smaller on
  ZuCo. Neutral reviews sit at the *highest* mean nFix in the z-scored full
  SST table. [`docs/findings.md`](docs/findings.md)
- **Two feature spaces.** `result/*.png` histograms are in raw
  millisecond/count space (`gaze_prediction/data/`). The SST training CSVs
  are z-scored (mean ≈ 0, std ≈ 1). Do not mix them.

## Layout

```
model_full_SST.py          full-SST trainer (RoBERTa + sidecar)
model_ZuCo_SST.py         400-row ZuCo trainer (k-fold)
utils_ZuCo.py             MATLAB .mat → sentence/word tables
SST_data/                 11.8k reviews + z-scored predicted gaze
ZuCo_SST_data/            400 actually-read reviews
ZuCo_et_csv_data/         12 readers, sentence- and word-level
gaze_prediction/data/    predictor I/O (zeros, Provo, predicted slices)
result/                   scatter/hist PNG grids (raw feature space)
docs/                     personal atlas notes
examples/                 CPU atlas scripts + sidecar/ library
tests/                    unittest (no Hub, no GPU)
```

## Training the original scripts

That path needs `requirements-train.txt`, Hugging Face weights, and (for the
full SST run) a GPU. The trainers write `models/best_{model_type}_model.pth`
and never `mkdir` that folder. See [`docs/scripts-as-found.md`](docs/scripts-as-found.md).

## Personal scope

This repository is a personal project (ZuCo + SST movie reviews). The atlas
does not include any company code.
