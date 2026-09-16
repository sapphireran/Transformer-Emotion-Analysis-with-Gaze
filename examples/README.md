# Examples

Runnable companions to [`docs/`](../docs/README.md). None of these
scripts download Hugging Face weights or train BERT/RoBERTa. They only
read the CSVs already in the tree.

Install the small dependency set first:

```bash
python -m pip install -r requirements.txt
```

Run every script from the **repository root** (or rely on `paths.py`,
which locates the root from `__file__` either way):

```bash
python examples/inspect_datasets.py
python examples/data_integrity.py
python examples/gaze_feature_report.py
python examples/gaze_only_baseline.py
python examples/late_fusion_demo.py
python examples/word_level_preview.py
python examples/sentence_walkthrough.py

python examples/run_all.py
```

`run_all.py` stops with a non-zero exit code if any child fails and
writes a transcript to `examples/output/run_all.txt`. A snapshot of the
numbers from the first green run is in
[`docs/example-results.md`](../docs/example-results.md).

## What each script is for

| Script | Reads | Prints / checks |
|---|---|---|
| `inspect_datasets.py` | every training and ET CSV listed in `paths.py` | row counts, columns, sentiment label histograms |
| `data_integrity.py` | SST + ZuCo splits, per-subject files | partition / alignment / NaN / subject-3 length assertions |
| `gaze_feature_report.py` | ZuCo standard join + full-SST train | per-feature stats, per-class means, Pearson *r* vs label |
| `gaze_only_baseline.py` | ZuCo standard join; full-SST train/test | logistic regression vs majority-class dummy |
| `late_fusion_demo.py` | ZuCo standard join | hashed bag-of-words ± Linear(5→16) softmax, 5-fold |
| `word_level_preview.py` | word-level ZuCo, predicted SST gaze, PROVO | tokens / sentences / feature ranges |
| `sentence_walkthrough.py` | ZuCo standard join + `word_averages_v2.csv` | one sentence per class, 5-d vector + word table |

## How this maps to the training scripts

```
inspect_datasets.py      →  "what files exist and how big are they?"
data_integrity.py        →  "did spilt.py actually partition the join?"
gaze_feature_report.py   →  "do the 5 ET columns move with polarity?"
gaze_only_baseline.py    →  "is there a linear ET-only signal?"
late_fusion_demo.py      →  "does a tiny late-fusion head behave like EyeTrackingModel?"
word_level_preview.py    →  "what do the gaze-predictor tables look like?"
sentence_walkthrough.py  →  "what does one training row contain, word by word?"
```

`late_fusion_demo.py` is a **didactic** stand-in: 64-d hashed unigrams
instead of a 768-d `pooler_output`, ReLU on the 5→16 map (the original
`EyeTrackingModel` has no activation), trained with numpy SGD. Use it to
understand the concat head, not to quote numbers against RoBERTa.

## Adding another check

1. Put the script next to these files.
2. Import `REPO_ROOT` (and the directory constants) from `paths.py`.
3. Append the filename to `SCRIPTS` in `run_all.py`.
4. Mention it in the table above and in `docs/README.md`.
