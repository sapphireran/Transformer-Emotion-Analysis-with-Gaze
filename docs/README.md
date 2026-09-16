# Documentation index

These notes describe the **personal** Transformer + gaze sentiment project in
this repository. They are written against the code and CSVs that are actually
in the tree, not against a cleaned-up paper reproduction.

Start here if you are coming back to the repo after a break.

| Note | What it answers |
| --- | --- |
| [overview.md](overview.md) | Why two tracks (human ZuCo vs predicted full SST) exist |
| [file-map.md](file-map.md) | Every directory and the scripts that write/read it |
| [data-pipeline.md](data-pipeline.md) | MATLAB → CSV → join SST labels → train tables |
| [gaze-features.md](gaze-features.md) | FFD, GD, TRT, GPT, SFD, nFixations, omission, pupil |
| [model-architecture.md](model-architecture.md) | Fusion head, tokenizers, loss, CV vs hold-out |
| [experiments.md](experiments.md) | Hyperparameters as hard-coded in the two model files |
| [known-pitfalls.md](known-pitfalls.md) | Path mismatches, metric bugs, naming traps |
| [reproducing.md](reproducing.md) | What you can run from CSVs vs what needs `.mat` / GPU |
| [citations.md](citations.md) | Dataset and model papers |

Runnable walkthroughs live in [`../examples/`](../examples/README.md). They
only need the CSVs plus the Python standard library and `numpy`.
