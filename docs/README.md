# Documentation index

These notes describe the personal ZuCo + SST experiments in this repository.
They are meant to be read next to the original scripts, not as a replacement
for the public ZuCo / SST papers.

| Page | What it covers |
| --- | --- |
| [overview.md](overview.md) | Why gaze is fused with a transformer, and the two tracks |
| [gaze-features.md](gaze-features.md) | Definitions of nFix, FFD, GPT, TRT, GD, SFD, pupil, omission |
| [datasets.md](datasets.md) | Every checked-in CSV, column lists, row counts, label balance |
| [data-pipeline.md](data-pipeline.md) | MATLAB → per-subject CSV → averages → SST join → splits |
| [models-and-training.md](models-and-training.md) | EyeTrackingModel, CV vs hold-out, metrics |
| [reproducing-experiments.md](reproducing-experiments.md) | Commands, expected artifacts, hardware notes |
| [known-issues.md](known-issues.md) | Path mismatches, test-loop overwrite, comment drift |

Runnable companions live in [`../examples`](../examples/README.md).
