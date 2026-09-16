# Personal examples

Runnable companions for the ZuCo + SST gaze experiments. No Hugging Face weights, no MATLAB, no company data.

| Script | Writes |
|---|---|
| `01_inspect_datasets.py` | `output/dataset_inventory.md`, `.html` |
| `02_gaze_feature_report.py` | `output/gaze_feature_report.md`, `.png` |
| `03_text_vs_gaze_baselines.py` | `output/baseline_comparison.md`, `.html`, `.json` |
| `04_sentence_walkthrough.py` | `output/sentence_walkthrough.md` |
| `05_full_sst_sample.py` | `output/full_sst_sample.md` |

From the repo root:

```bash
pip install -r requirements.txt
PYTHONPATH=. python3 examples/01_inspect_datasets.py
PYTHONPATH=. python3 examples/run_all.py
```

`tea_gaze` is the importable layer (paths, schemas, metrics, logistic baselines). The original `model_*.py` scripts stay standalone.
