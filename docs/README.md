# Documentation index

Personal notes for this research repo. Read these in order if you are
coming back to the project after a break.

| Doc | What it covers |
| --- | --- |
| [overview.md](overview.md) | Research question, the two tracks, and how the pieces fit. |
| [datasets.md](datasets.md) | Every CSV: shape, columns, label balance, how it was produced. |
| [eye_tracking_features.md](eye_tracking_features.md) | What nFix / FFD / GPT / TRT / GD / SFD actually measure. |
| [models.md](models.md) | BERT/RoBERTa baselines and the `EyeTrackingModel` fusion head. |
| [data_pipeline.md](data_pipeline.md) | Script-by-script path from ZuCo `.mat` / SST text to training CSVs. |
| [training_and_evaluation.md](training_and_evaluation.md) | Hyperparameters, splits, metrics, and how to resume a run. |
| [known_issues.md](known_issues.md) | Bugs and path mismatches in the original scripts. |
| [glossary.md](glossary.md) | Short definitions used throughout the notes. |
| [reproducing_local_analysis.md](reproducing_local_analysis.md) | How to rerun the `examples/` suite without a GPU. |

Runnable companions live in [`examples/`](../examples/README.md).
