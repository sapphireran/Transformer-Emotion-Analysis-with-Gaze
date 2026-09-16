# Personal examples

Small scripts that read the CSVs already in this repository. They do **not**
download BERT, start a GPU job, or call any network API. Use them to see
what the tables contain before editing `model_ZuCo_SST.py` or `model_full_SST.py`.

Run every command from the **repository root** so `from common import ...`
resolves (`sys.path[0]` is `examples/` when you invoke `python3 examples/…`).

```bash
python3 examples/run_all.py
```

or one at a time:

| Script | What it prints |
| --- | --- |
| `inspect_datasets.py` | File inventory vs the documented row counts, including the short subject-3 tables |
| `label_distribution.py` | 3-class counts for every labeled CSV, plus a reminder that the ZuCo 80/10/10 split is unstratified |
| `feature_stats.py` | min / mean / max / zero-rate for gaze columns, plus Pearson correlations on ZuCo |
| `schema_validate.py` | headers, labels ∈ {0,1,2}, finite gaze; exit 0 only if everything matches |
| `sentence_gaze_join.py` | Rebuilds text ⨝ averaged gaze and diffs it against the combined CSVs |
| `sample_rows.py` | One ZuCo sentence per class, SST length extremes, a word window |
| `word_level_preview.py` | Skip rates, PROVO size, placeholder vs predicted SST word tables |
| `gaze_fusion_demo.py` | 5 → 16 concat 768 → 3 shape sketch and a shuffle-gaze ablation |

`common.py` is the shared inventory (`DATASETS`), path helper, and stats
functions. It is not a public package.

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | ok |
| 1 | soft mismatch (row count, join drift) |
| 2 | missing file or broken required column |

`run_all.py` stops at the first non-zero exit.

## What these are not

- Not unit tests of `EyeTrackingModel` (that needs PyTorch).
- Not a replacement for `model_*.py`.
- Not company fixtures. Paths and counts are specific to this personal clone.

After a GPU run, keep metric JSON under `result/` rather than extending these
scripts into a trainer.
