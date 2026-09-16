# Personal write-up index

These notes describe the checked-in tables and the two training scripts in
this personal repository. They are meant to be read next to the CSVs, not
as a substitute for the ZuCo or SST papers.

## Reading order

1. **[datasets.md](datasets.md)** — inventory of every CSV, with row counts
   and the label split you should expect if a file was not edited.
2. **[gaze-features.md](gaze-features.md)** — what each eye-tracking column
   measures, how sentence-level values are aggregated, and why some cells
   are zero.
3. **[pipeline.md](pipeline.md)** — the extraction path from ZuCo `.mat`
   files through subject averages to the two experiment tables.
4. **[model-architecture.md](model-architecture.md)** — encoder choices,
   the late-fusion gaze MLP, tokenization, and the four `model_type` flags.
5. **[experiments.md](experiments.md)** — hyperparameters, folds, and what
   “best model” actually tracks in each script.
6. **[known-issues.md](known-issues.md)** — historical bugs left in the
   original trainers (test-loop overwrite, path separators, comments that
   say F1 when the code saves accuracy).
7. **[glossary.md](glossary.md)** — short definitions.
8. **[reproducing-locally.md](reproducing-locally.md)** — what you can run
   without MATLAB, without GPU, and without re-downloading SST.
9. **[design-notes.md](design-notes.md)** — why late fusion, why five
   features on full SST, and what this setup does *not* claim.
10. **[worked-example.md](worked-example.md)** — one ZuCo sentence, word by
    word, then the sentence-level vector the classifier actually sees.

## Companion programs

Runnable counterparts live under [`../examples/`](../examples/README.md).
Each doc that cites a number (400 sentences, 12 subjects, 191,971 predicted
tokens) can be re-checked with `examples/inspect_datasets.py`.

## What this write-up is not

- Not a re-release of ZuCo or SST.
- Not a claim that gaze *causes* sentiment.
- Not a cleaned production training library. The trainers are research
  scripts; the examples are the maintained personal layer.
