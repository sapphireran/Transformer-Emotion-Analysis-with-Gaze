# Personal examples

Small programs that sit on top of the checked-in CSVs. They do not
download Hub weights, do not need a GPU, and do not call the historical
trainers.

Install the light stack from the repo root:

```bash
python3 -m pip install -r requirements-examples.txt
```

Then run any script as `python3 examples/<name>.py`.

## Inventory and sanity

| Script | What it does |
| --- | --- |
| `inspect_datasets.py` | Row counts, headers, joins, 12×400 / 12×7129 subject files |
| `compare_scalings.py` | Min-max vs z-score tables are the same 400 sentences |
| `split_balance.py` | Label priors on every official split |
| `gaze_feature_stats.py` | Min/mean/max + Pearson matrices, measured vs projected |

```bash
python3 examples/inspect_datasets.py --strict
python3 examples/compare_scalings.py
python3 examples/split_balance.py
python3 examples/gaze_feature_stats.py --out examples/output
```

`--strict` makes the inspector exit 1 on a mismatch. Tests use that.

## Looking at one sentence

| Script | What it does |
| --- | --- |
| `sentence_gaze_walkthrough.py` | Text, word table, two subjects, fusion z-scores |
| `word_level_skip_analysis.py` | Skip rates per subject; leftover zeros after averaging |
| `subject_variability.py` | Between- vs within-sentence spread across 12 readers |

```bash
python3 examples/sentence_gaze_walkthrough.py --sentence-id 3
python3 examples/word_level_skip_analysis.py
python3 examples/subject_variability.py --sentence-id 3
```

Sentence 3 is the write-up in `docs/worked-example.md`.

## Cheap late-fusion stand-in

`toy_text_gaze_fusion.py` runs stratified 5-fold logistic regression
with four feature blocks: length, gaze-only, hashed text, text+gaze.
It answers “is there any linear signal in the five z-scores?” — not
“does RoBERTa improve”.

```bash
python3 examples/toy_text_gaze_fusion.py --folds 5
```

## Library

`examples/lib/` is the shared layer:

- `paths.py` — inventory of canonical files and expected row counts
- `loaders.py` — headerless SST, fusion-column rename, word-row lookup
- `metrics.py` — accuracy, weighted P/R/F1, **macro** F1
- `fusion.py` — hashed-text pipelines and `run_fusion_cv`

Tests import the same library (`tests/`).

## Output directory

`--out examples/output` writes text reports. That folder is gitignored.
Checked-in snapshots, when present, live under `docs/sample-reports/`.
