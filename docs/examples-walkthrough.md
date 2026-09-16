# Examples walkthrough

A personal lab-notebook path through `examples/`. The scripts do not train
transformers. They tell you whether the committed tables are internally
consistent and whether sentence-level gaze has any linear sentiment signal.

## 0. Install

```bash
python3 -m pip install -r requirements.txt
```

`pandas`, `numpy`, and `scikit-learn` (which pulls in `scipy`) are enough.

## 1. Inventory check

```bash
python3 examples/inspect_datasets.py
```

This is the gate. It fails if:

- a subject CSV is missing
- train/valid/test ids overlap or drop a class
- a fusion column is non-numeric
- PROVO suddenly grows a `GD` column (the inventory says `fixProp`)

If this script fails, do not train. Fix the table first.

## 2. Rebuild the ZuCo join

```bash
python3 examples/join_zuco_sst.py
```

Inner-joins `ssts_ZuCo.csv` onto the subject-averaged, scaled gaze tables
and diffs fusion columns against `combined_sst_et_*.csv`. A max absolute
difference above `1e-6` means someone edited one side of the join by hand.

## 3. Subject disagreement

```bash
python3 examples/subject_agreement.py
```

The classifier never sees individual readers. If TRT median CV is large,
the five-dimensional gaze vector is an average of unlike behaviors. Look
at the leave-one-out correlations: a single low-r subject can dominate
skip-rate.

## 4. Class-conditional gaze

```bash
python3 examples/gaze_feature_stats.py --write examples/sample_outputs/gaze_feature_stats.json
```

Read `f_classif` before you read means. On ~400 ZuCo rows a pretty mean
shift on the neutral class is often noise. On full SST, projected gaze
should look weaker; if it looks *stronger* than real ZuCo gaze, the
projection model may have leaked the label.

## 5. Linear floors

```bash
python3 examples/sentiment_baselines.py --write examples/sample_outputs/baselines.json
python3 examples/fusion_toy.py --write examples/sample_outputs/fusion_toy.json
```

Interpretation cheatsheet:

| Pattern | Meaning |
| --- | --- |
| gaze ≈ chance, tfidf ≫ gaze | eyes are not a standalone classifier; good |
| tfidf+gaze ≈ tfidf | concat residual is weak; GPU fusion must work non-linearly or not at all |
| tfidf+gaze ≫ tfidf on ZuCo, not on full SST | real gaze has signal that the projector did not transfer |
| length ≈ gaze | you may be measuring sentence length, not reading time |

`fusion_toy.py` also projects the five gaze features to 16 dimensions with
a fixed random matrix so the concat width matches `EyeTrackingModel`'s
`hidden_layer_size = 16`.

## 6. Word-level sanity

```bash
python3 examples/word_level_gaze.py --write examples/sample_outputs/word_level_gaze.json
```

Checks:

- `word_id` / `Word_ID` is 0..n-1 per sentence
- no non-finite durations
- long alphabetic tokens attract more fixations than a small function-word list
- predicted v2 sentence ids vs. `combined_full_sst_et.csv`

A gap between predicted ids and SST ids is not automatically a bug. The
sentence-level full-SST table may have been aggregated from a different
prediction dump. Record the overlap and move on.

## 7. After you change a CSV

Re-run `inspect_datasets.py` and `join_zuco_sst.py`. Both are exit-code
sensitive. Then refresh the JSON dumps if you keep them in git.

## 8. After you change a training script

The examples do not import `model_ZuCo_SST.py` or `model_full_SST.py`.
Update [architecture.md](architecture.md) by hand if you change
`model_type`, column lists, or the test loop.
