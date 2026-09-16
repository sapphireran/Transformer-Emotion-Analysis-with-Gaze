# Documentation index

These notes describe the **personal** Transformer + gaze sentiment
experiments in this repository. They are written from the checked-in
scripts and CSVs, not from a paper.

| Note | What it covers |
|---|---|
| [datasets.md](datasets.md) | Every CSV: columns, row counts, label balance, which files feed which script |
| [eye-tracking-features.md](eye-tracking-features.md) | What nFix / FFD / GPT / TRT / GD / SFD actually measure |
| [data-pipeline.md](data-pipeline.md) | End-to-end path from ZuCo `.mat` and SST text to model tensors |
| [model-architecture.md](model-architecture.md) | Late-fusion `EyeTrackingModel`, baselines, loss, metrics |
| [reproduction.md](reproduction.md) | Commands, expected artifacts, hardware notes |
| [known-quirks.md](known-quirks.md) | Bugs and path mismatches to be aware of before you re-run |
| [example-results.md](example-results.md) | Snapshot of integrity / baseline / toy-fusion numbers |

Runnable companions live under [`../examples/`](../examples/README.md):

- inventory + label counts
- schema / NaN / split integrity
- gaze-only logistic baseline vs majority class
- hashed-bag-of-words + gaze late-fusion toy model
- word-level file preview (ZuCo averages, predicted SST gaze, PROVO)
- one-sentence-per-class walkthrough (text, 5-d vector, word ET)
