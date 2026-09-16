# Examples

Runnable walkthroughs of the CSVs already in this repository. They document
the two experiment tracks (human ZuCo gaze vs predicted full SST) **without**
fine-tuning BERT/RoBERTa.

Need Python 3.10+ and `numpy`. From the repository root:

```bash
python examples/inspect_sentence_gaze.py
python examples/inspect_word_gaze.py
python examples/subject_coverage.py
python examples/label_and_gaze_summary.py
python examples/sst_split_sanity.py
python examples/predicted_gaze_preview.py
python examples/toy_fusion_forward.py
python examples/gaze_only_baseline.py
python examples/metrics_example.py
```

Each script inserts the repo root on `sys.path` so `python examples/foo.py`
works. You can also use `python -m examples.inspect_sentence_gaze`.

## What each script is for

| Script | Reads | Tells you |
| --- | --- | --- |
| `inspect_sentence_gaze.py` | `ZuCo_et_csv_data/{1–12}_SR.csv`, averages, scaled averages | Row counts per reader; raw vs z-score vs min-max ranges |
| `inspect_word_gaze.py` | `ZuCo_et_csv_data/word/` | Skip rate, TRT vs GD, WordLen–nFixations correlation |
| `subject_coverage.py` | sentence + word subject files | Subject 3's 100-sentence hole; how averaging aligns on `id` |
| `label_and_gaze_summary.py` | `ZuCo_SST_data/combined_sst_et_standard.csv` and the 80/10/10 split | Class balance; class-conditional gaze means; which features carry label signal |
| `sst_split_sanity.py` | ZuCo combined + full SST splits + `stts_all_sentence_level.csv` | Disjoint IDs; label mapping; **do not join Track A/B on sentence_id** |
| `predicted_gaze_preview.py` | `SST_data/*_full_sst.csv`, `gaze_prediction/data/` | Predicted gaze can be negative / >1; PROVO has `fixProp` not `GD` |
| `toy_fusion_forward.py` | none (random tensors) | Shapes `(B,768) ⊕ (B,16) → (B,784) → (B,3)` plus a deterministic concat check |
| `gaze_only_baseline.py` | Track A combined table | 5-fold softmax on majority / length / gaze / gaze+length |
| `metrics_example.py` | synthetic batches | Weighted vs macro F1; the last-batch overwrite in `model_full_SST.py` |

`examples/common.py` holds CSV loading, the fusion column order, stratified
fold indices, and the classification report used by the baseline and metric
scripts.

## Suggested order the first time

1. `sst_split_sanity.py` — confirm the two ID spaces and split sizes.
2. `subject_coverage.py` — do not claim N=12 on every averaged sentence.
3. `inspect_sentence_gaze.py` / `inspect_word_gaze.py` — units and skip rate.
4. `label_and_gaze_summary.py` — is there any linear gaze–label structure?
5. `gaze_only_baseline.py` — numeric floor for Track A fusion.
6. `toy_fusion_forward.py` — what the GPU model is concatenating.
7. `predicted_gaze_preview.py` — before a Track B GPU run.
8. `metrics_example.py` — how to quote numbers from the original training loops.

## What these examples deliberately do *not* do

- Download `bert-base-uncased` / `roberta-base`
- Call `utils_ZuCo.DataTransformer` (needs MATLAB `.mat` files)
- Write under `models/`
- Treat predicted gaze as milliseconds

Training remains `model_ZuCo_SST.py` and `model_full_SST.py` with
`requirements.txt`. Pitfalls: [`docs/known-pitfalls.md`](../docs/known-pitfalls.md).
