# Personal research notes

These notes describe the files already in this repository. They are not a
company playbook and they do not depend on any private workplace codebase.

Start here, then jump to the page that matches the question:

| Note | What it is for |
| --- | --- |
| [datasets.md](datasets.md) | Which CSVs exist, how many rows, which split script made them |
| [data-dictionary.md](data-dictionary.md) | Column-by-column meanings and type checks |
| [gaze-features.md](gaze-features.md) | nFix, FFD, GD, TRT, GPT, SFD, pupil, omission rate |
| [architecture.md](architecture.md) | How text and eye-tracking vectors are concatenated |
| [model-variants.md](model-variants.md) | `bert` / `roberta` vs the `_eye_tracking` variants |
| [pipeline.md](pipeline.md) | MATLAB → CSV → join → train, in the order the scripts run |
| [reproduction.md](reproduction.md) | Commands I actually use on this laptop |
| [experiment-notes.md](experiment-notes.md) | Numbers computed from the checked-in tables |
| [known-issues.md](known-issues.md) | Bugs and naming traps in the original scripts |
| [citations.md](citations.md) | Public datasets and papers this personal project leans on |
| [example-output.md](example-output.md) | Invariants the example scripts should keep printing |

Runnable walkthroughs live in [`../examples/`](../examples/README.md). Shared
column names and loaders live in [`../tea_gaze/`](../tea_gaze/).
