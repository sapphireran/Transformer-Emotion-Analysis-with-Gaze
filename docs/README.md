# Documentation index

Personal notes for this repository. Nothing here trains a transformer; that still lives in the two `model_*.py` scripts. These pages describe **what the committed CSVs contain**, **how they were built**, and **where the original code is easy to misread**.

| Page | Contents |
| --- | --- |
| [file-map.md](file-map.md) | Every top-level script and data directory |
| [datasets.md](datasets.md) | ZuCo Task 1, SST, Provo, predicted gaze tables |
| [gaze-features.md](gaze-features.md) | Definitions, units, scaling, collinearity |
| [data-pipeline.md](data-pipeline.md) | MAT → subject CSV → average → join → split |
| [architecture.md](architecture.md) | BERT/RoBERTa + gaze concat |
| [training.md](training.md) | Hyperparameters, metrics, checkpoints |
| [known-issues.md](known-issues.md) | Test-loop bug, subject-3 alignment, paths |
| [references.md](references.md) | Papers and corpora to cite |
| [examples-walkthrough.md](examples-walkthrough.md) | CPU example scripts |

Generated figures from the example scripts are stored in [assets/](assets/). Historical scatter plots from earlier work remain in [`../result/`](../result/).

## Reading order

1. Skim the root [README](../README.md) for the two tracks.
2. Use [datasets.md](datasets.md) when opening a CSV so column names do not get mixed (`nFix` vs `nFixations`, `fixProp` vs `GD`).
3. Read [known-issues.md](known-issues.md) before averaging subjects or quoting `model_full_SST.py` test accuracy.
4. Run `make test` and `make examples` from the repo root (see [../examples/README.md](../examples/README.md)).
