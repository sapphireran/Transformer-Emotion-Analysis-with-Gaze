# Personal docs

Reading notes for this clone. The original training scripts are left
alone. Numbers in the generated notebook come from the stdlib examples.

| page | start here if you want |
| --- | --- |
| [Project map](project-map.md) | A tour of what is actually in the tree |
| [Datasets](datasets.md) | The two tracks, row counts, label balance |
| [Feature dictionary](feature-dictionary.md) | nFixations, FFD, GD, GPT, TRT, … |
| [Subject alignment](subject-alignment.md) | Reader 3 is remapped after id 149 |
| [Pipeline](pipeline.md) | MATLAB → CSV → z-score → model input |
| [Models](models.md) | BERT/RoBERTa ± Linear(5→16) concat |
| [Known bugs](known-bugs.md) | Test-loop overwrite, path leftovers |
| [Hypotheses](hypotheses.md) | What gaze could add, and what it probably does not |
| [Glossary](glossary.md) | Short definitions |
| [Generated notebook](generated/lab_notebook.md) | Fresh tables from `examples/run_all.py` |

Suggested order: map → subject alignment → known bugs → datasets → models.
