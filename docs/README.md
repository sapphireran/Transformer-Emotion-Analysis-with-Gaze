# Docs index

Personal notes for `Transformer-Emotion-Analysis-with-Gaze`. They describe the
code and CSVs that are actually in this repository, including the messy parts.

| Note | Contents |
| --- | --- |
| [overview.md](overview.md) | Research question, two tracks, what "emotion" means here |
| [datasets.md](datasets.md) | File inventory, schemas, row counts, label mix |
| [gaze-features.md](gaze-features.md) | Reading-measure definitions and collinearity |
| [preprocessing.md](preprocessing.md) | MATLAB extract, subject average, SST join, scaling |
| [architecture.md](architecture.md) | BERT/RoBERTa + linear gaze fusion |
| [training-and-evaluation.md](training-and-evaluation.md) | Hyperparameters, CV, metrics, eval pitfalls |
| [reproduction.md](reproduction.md) | Commands that work from a fresh clone |
| [glossary.md](glossary.md) | Short glossary |
| [notes/known-limitations.md](notes/known-limitations.md) | Bugs and alignment issues left in the original scripts |
| [notes/worked-example-sentence-0.md](notes/worked-example-sentence-0.md) | One sentence, word-level gaze |
| [notes/linear-baselines.md](notes/linear-baselines.md) | TF-IDF ± gaze 5-fold numbers |
| [generated/dataset-report.md](generated/dataset-report.md) | Auto-generated tables from the CSVs |

Runnable companions live in [`../examples`](../examples).
