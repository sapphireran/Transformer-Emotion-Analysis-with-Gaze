# Personal examples

Stdlib-only scripts that read the checked-in CSVs. They do not train BERT or
RoBERTa and they do not rewrite the original experiment files.

Run from the repository root (or from this folder; `_util.py` puts the repo on
`sys.path`):

```bash
python3 examples/01_repo_inventory.py
python3 examples/02_subject_alignment.py
python3 examples/03_reader_agreement.py
python3 examples/04_label_conditioned_gaze.py
python3 examples/05_word_sentence_bridge.py
python3 examples/06_split_and_schema.py
python3 examples/07_test_loop_bug.py
python3 examples/08_cpu_fusion.py
python3 examples/09_gaze_baselines.py
python3 examples/10_provo_preview.py
```

Write markdown snapshots and the combined notebook:

```bash
python3 examples/run_all.py
```

Snapshots land in `examples/outputs/` and a copy of the notebook is also
written to `docs/generated/`.

| script | what it is for |
| --- | --- |
| `01_repo_inventory.py` | Consumer tables, row counts, schema flags |
| `02_subject_alignment.py` | Reader 3 remapping and average contamination |
| `03_reader_agreement.py` | CV / pairwise r on the clean 0–149 window |
| `04_label_conditioned_gaze.py` | Gaze means by sentiment, collinearity |
| `05_word_sentence_bridge.py` | One sentence from word averages to the train row |
| `06_split_and_schema.py` | 320/40/40 and 80/10/10 integrity |
| `07_test_loop_bug.py` | Full-SST test loop keeps the last batch |
| `08_cpu_fusion.py` | Concat(text, Linear(5→16)) wiring demo |
| `09_gaze_baselines.py` | Majority vs gaze-only least squares |
| `10_provo_preview.py` | PROVO vs predicted word-level tables |
| `11_write_lab_notebook.py` | One markdown + HTML notebook |

Helpers live in `zuco_lab/`. Tests live in `tests/`.
