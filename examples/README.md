# Examples

These scripts only read the CSVs that are already in the repository. They
do not download BERT/RoBERTa and they do not need a GPU.

Run any file from the **repository root** so relative data paths resolve:

```bash
python3 examples/inspect_datasets.py
python3 examples/gaze_feature_tour.py
python3 examples/label_and_split_audit.py
python3 examples/fusion_architecture_demo.py
python3 examples/synthetic_training_loop.py
```

Or everything, plus pytest:

```bash
bash examples/run_all.sh
```

| Script | What it shows |
| --- | --- |
| `inspect_datasets.py` | Row counts, columns, label balance, gaze means for every known table |
| `gaze_feature_tour.py` | Definitions plus extreme ZuCo sentences for each fused channel |
| `label_and_split_audit.py` | 80/10/10 leakage check, subject 3 coverage, majority baselines |
| `fusion_architecture_demo.py` | Shapes of the late-fusion classifier, one forward pass |
| `synthetic_training_loop.py` | Tiny SGD on separable text+gaze batches; text-only vs fusion |

Shared helpers live in `gaze_emotion/`. Edit those modules if you want the
same numbers in a notebook.
