# Personal examples (CPU only)

These scripts do **not** train BERT/RoBERTa. They read the committed CSVs and
write markdown/HTML under `examples/outputs/`.

```bash
python3 -m pip install -r requirements.txt
python3 examples/run_all.py
python3 -m unittest discover -s tests -v
```

| Script | What it pins down |
| --- | --- |
| `01_inventory.py` | Every table path the atlas expects |
| `02_label_atlas.py` | Label priors, majority baselines, r(gaze, label) |
| `03_sidecar_rank.py` | PCA: 5 gaze columns ≈ 1–2 directions |
| `04_subject3_reindex.py` | Packed ids: CSV row 150 is original sentence 250 |
| `05_split_fingerprint.py` | Id-disjoint splits, two leaked review strings |
| `06_last_batch_metric.py` | `model_full_SST.py` test loop keeps the last batch only |
| `07_length_confound.py` | Predicted nFix anti-correlates with token count |
| `08_fusion_forward.py` | NumPy 768+16 concat (no Hub download) |
| `09_word_skips.py` | Per-reader skip rates; subject 3 is the outlier |
| `10_shuffle_control.py` | Permute gaze, watch class means collapse |
| `11_gaze_prediction_track.py` | Raw vs z-scored feature spaces |
| `12_reader_means.py` | Pupil / omission / TRT by reader |
| `13_write_html_atlas.py` | Combines the markdown into `sidecar_atlas.html` |

Importable helpers live in `examples/sidecar/`. Original 2024 training
scripts stay at the repo root; they need `requirements-train.txt` and a GPU
if you actually want to fit RoBERTa.
