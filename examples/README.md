# Examples

GPU-free scripts that read the personal CSVs in this clone. They share
`gaze_emotion_examples/`, a small library with a catalog, stats helpers, a
numpy fusion sketch, and sklearn text-vs-gaze baselines.

```bash
python -m pip install -r requirements-examples.txt
python examples/01_inspect_catalog.py
python -m pytest
```

Run every script from the **repository root** so relative imports and
`repo_root()` resolve.

## Scripts

### 01 — inspect the catalog

Opens every entry in `gaze_emotion_examples/catalog.py` and prints key,
kind, row count, and path. Fails loudly if a declared column is missing.

### 02 — sentence-level gaze stats

Descriptive tables, label-conditional means, high Pearson pairs, and a
correlation heatmap under `examples/output/corr_<key>.png`. Default keys:
`zuco_sst_standard`, `zuco_sentence_raw`, `sst_full`.

```bash
python examples/02_sentence_gaze_stats.py --keys zuco_sst_standard
```

### 03 — label balance

Prints SST-3 counts for both tracks and their stored splits, plus the
majority-class accuracy. Writes `examples/output/label_balance.png` and
`label_balance.csv`.

### 04 — compare scalers

Confirms that `combined_sst_et_standard.csv` is z-score-like,
`combined_sst_et_min_max.csv` is `[0, 1]`, and the raw average table is
neither. Prints Spearman rank agreement of each gaze column against the raw
table (monotone column scaling should stay at 1.0).

### 05 — word-level profile

Rebuilds one ZuCo sentence from `word_averages_v2.csv`, ranks tokens by TRT
and GPT, and plots total reading time.

```bash
python examples/05_word_level_profiles.py --sentence-id 0_NR
```

See `docs/notes/worked-example-sentence-0.md` for the commentary.

### 06 — text vs gaze baselines

Prints the `(B, 768) + (B, 5) → (B, 3)` shape sketch, then runs 5-fold
stratified logistic regression:

| model | features |
| --- | --- |
| `gaze_only` | 5 sentence gaze channels |
| `text_only` | TF-IDF 1–2 grams |
| `text_plus_gaze` | concatenation of the two |

Writes `examples/output/text_vs_gaze_baselines.csv`. This is a linear probe,
not RoBERTa.

```bash
python examples/06_text_vs_gaze_baselines.py --gaze-columns nFixations TRT FFD
```

### 07 — split audit

Checks that ZuCo and full-SST 80/10/10 files are disjoint and cover their
parent tables. Exits non-zero on leakage. Also prints label-mix drift
(ZuCo valid is not stratified).

### 08 — dataset report

Rewrites `docs/generated/dataset-report.md` from the CSVs. Commit the result
when the data changes.

## Library map

| Module | Responsibility |
| --- | --- |
| `catalog.py` | `DatasetSpec` list and `get_dataset` |
| `io.py` | `load_dataset` with column checks |
| `labels.py` | 0/1/2 ↔ negative/neutral/positive |
| `stats.py` | describe, correlate, markdown tables |
| `scaling.py` | standard vs min-max fingerprints |
| `splits.py` | ID leakage audit |
| `profiles.py` | word-level `WordGaze` rows |
| `fusion.py` | numpy `EyeTrackingModel` + sklearn CV |
| `reports.py` | markdown report builder |
| `paths.py` | `repo_root()` |

## Tests

`tests/` imports the same package (see `pytest.ini` `pythonpath = examples`).
The tests load real CSVs for catalog / scaling / split / profile checks and
use tiny synthetic frames for fusion math.
