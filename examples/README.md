# Examples

CPU-only scripts that **read the committed CSVs** and write markdown/JSON under
`examples/output/` plus a few figures under `docs/assets/`. They do not train
BERT/RoBERTa and they do not need MATLAB.

```bash
python3 -m pip install -r requirements-examples.txt
# from the repository root
export PYTHONPATH=examples
python3 -m pytest examples/tests
python3 examples/scripts/run_all.py
# or
make test
make examples
```

## Package `teag_examples`

Importable helpers (Transformer Emotion Analysis with Gaze):

| Module | Responsibility |
| --- | --- |
| `paths` | find repo root from any cwd |
| `schema` | column names, label maps, expected row counts |
| `io` | load + validate each CSV |
| `stats` | row counts, missingness, label histograms |
| `gaze` | Pearson / collinear pairs / scaling checks |
| `pipeline` | reconstruct ZuCo combined tables |
| `alignment` | subject-3 compacted vs original sentence index |
| `splits` | disjoint `sentence_id` / coverage |
| `fusion` | NumPy `EyeTrackingModel` concat clone |
| `metrics` | weighted acc / P / R / F1 (same as training scripts) |
| `viz` | correlation heatmaps, label bars, fusion schematic |
| `reports` | markdown + JSON writers |

## Scripts

| Script | What it proves |
| --- | --- |
| `summarize_datasets.py` | inventory of ZuCo / SST / Provo / predicted-gaze tables |
| `analyze_gaze_features.py` | weak label correlation; SST predicted collinearity |
| `check_splits.py` | 80/10/10 files are disjoint and cover the combined table |
| `reconstruct_zuco_combined.py` | join(`ssts_ZuCo`, scaled averages) matches git |
| `check_subject_alignment.py` | `3_SR.csv` aligns only through original sentence 149 |
| `preview_word_gaze.py` | word-level ZuCo / predicted / Provo heads |
| `demo_fusion_forward.py` | 5→16 concat 784→3 with real gaze and random pooled text |
| `run_all.py` | runs the list above |

Each script is safe to re-run; outputs are overwritten.

## Tests

`examples/tests/` hits the committed data (not toy fixtures) for schema, join
reconstruction, split leakage, subject-3 index math, and the NumPy fusion
shapes. Fusion tests use a tiny `FusionConfig` so they do not depend on 768-D
noise beyond the demo script.

## What these examples deliberately skip

- Downloading `bert-base-uncased` / `roberta-base`
- GPU training
- Reading ZuCo `.mat` files
- Claiming that random-pooled fusion logits are sentiment accuracy
