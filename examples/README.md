# Examples (stdlib only)

These scripts reprint the numbers used in `docs/`. They do not train a
model and they do not import `pandas` or `torch`. Run them from the
**repository root** so the CSV paths resolve.

```bash
python examples/inspect_datasets.py
python examples/walk_sentence_gaze.py
python examples/walk_sentence_gaze.py --sentence-id 4
python examples/fusion_sketch.py
```

| Script | What it shows |
| --- | --- |
| `inspect_datasets.py` | Row counts, label histograms, gaze means/ranges for every training CSV |
| `walk_sentence_gaze.py` | One ZuCo sentence at word level, with a short commentary on peaks |
| `fusion_sketch.py` | Parameter count of the 5→16 and 784→3 layers; a numeric concat example using sentence 0's z-scores |

If `inspect_datasets.py` and `docs/00-reading-order.md` disagree, the
script wins. Update the doc.
