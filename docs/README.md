# Documentation index

Personal notes for this repository. Read the root [README](../README.md) first, then the pages below if you are reconstructing a run or adding a table.

## Start here

| Page | What it answers |
| --- | --- |
| [architecture.md](architecture.md) | How `EyeTrackingModel` is wired, and how the two training scripts differ |
| [data-dictionary.md](data-dictionary.md) | Column names, row counts, and which script reads each CSV |
| [data-pipeline.md](data-pipeline.md) | Order of conversion scripts from ZuCo MATLAB to joined training tables |
| [eye-tracking-features.md](eye-tracking-features.md) | What nFix / FFD / GPT / TRT / GD / SFD mean in this repo |
| [models-and-training.md](models-and-training.md) | Hyperparameters, metrics, checkpoints |
| [baselines.md](baselines.md) | Majority and gaze-only numbers from `examples/dummy_baseline.py` |
| [reproduction.md](reproduction.md) | Practical steps to re-run training or the example suite |
| [known-issues.md](known-issues.md) | Path mismatches, evaluation bugs, missing folders |
| [experiment-notes.md](experiment-notes.md) | Why certain defaults were chosen |
| [citations.md](citations.md) | Source corpora and suggested citations |

## How this docs tree relates to code

```
ZuCo .mat ──► utils_ZuCo.DataTransformer ──► ZuCo_et_csv_data/*_SR.csv
                                                │
                                                ▼
                                      subject average + scaling
                                                │
ssts_ZuCo.csv ────────────────────────► joined ZuCo_SST_data/*.csv
                                                │
                                                ▼
                                      model_ZuCo_SST.py (CV)

SST sentences ──► convert / join ──► SST_data/*_full_sst.csv
                                                │
                                                ▼
                                      model_full_SST.py (holdout)
```

The example scripts under `../examples/` only consume the already-joined CSVs. They are safe to run on a laptop without MATLAB, GPU, or Hugging Face downloads.

## Conventions used in these notes

- **Measured gaze** means values derived from ZuCo recordings (`ZuCo_et_csv_data/`).
- **Transferred / predicted gaze** means values attached to SST sentences that were not recorded with an eye tracker (`SST_data/`, `gaze_prediction/data/`).
- **Fusion features** always means the five-vector `(nFixations|nFix, FFD, GPT, TRT, GD)` in that order inside the PyTorch datasets.
- Sentiment is always `{0: negative, 1: neutral, 2: positive}` after the mapping in `convert_full_SST.py` / `save_SST_data.py`.

If a CSV and a script disagree, trust the script's column list and file this in [known-issues.md](known-issues.md).
