# Example walkthrough

This is the narrative companion to `examples/README.md`. It says what I look at first when I come back to this personal repo after a break.

## 1. Confirm the 400-row universe is intact

```bash
python3 examples/scripts/01_explore_zuco_sst.py
```

I want:

- 400 rows on both the combined table and `ssts_ZuCo.csv`
- matching `sentence_id` sets
- matching labels
- a majority-class fraction printed in the open (so later accuracies have a floor)

If the id sets diverge, the join that built `combined_sst_et_standard.csv` was edited. Stop and diff before training.

## 2. Look at the five channels as numbers, not as a story

```bash
python3 examples/scripts/02_gaze_feature_report.py
```

Checks:

- zero missing values on the fusion columns (the historical scaler filled them)
- TRT vs nFixations correlation high but not 1
- class-conditional means that are a fraction of a standard deviation apart

If `|Δmean| / std` is tiny on every channel, a linear fusion head has almost nothing to work with. That is still a valid negative result.

## 3. Spend five seconds on a model that is not BERT

```bash
python3 examples/scripts/03_gaze_only_baseline.py
```

Five stratified folds, `random_state=42` — the same fold recipe as `model_ZuCo_SST.py`, but the features are only gaze / length, and `StandardScaler` is fit inside the fold.

Interpretation cheatsheet:

| Pattern | Reading |
| --- | --- |
| majority ≈ gaze ≈ length | gaze is not a sentiment feature at sentence level |
| gaze > majority, length ≈ majority | maybe worth fusing |
| gaze ≈ length > majority | you are measuring sentence length |
| residualized gaze drops to majority | same, more formally |

The transformer fusion can still win when logistic loses, but I do not start a GPU job until I have seen this table.

## 4. Trust the split you are about to quote

```bash
python3 examples/scripts/04_split_sanity_check.py
```

`model_ZuCo_SST.py` **ignores** `train.csv` / `valid.csv` / `test.csv` and re-folds. I still run the checker because the example baselines and any future notebook will be tempted to use those files. They must be disjoint and cover all 400 ids.

The full SST checker is the one that matters for `model_full_SST.py`.

## 5. Remember these are eyes on words

```bash
python3 examples/scripts/05_word_level_scanpath.py
```

Word-length bins should show rising nFixations / GD / TRT. If they do not, the word average is broken (or zeros were treated as real fixations — see the commented `replace(0, nan)` in `word/get_average.py`).

The printed scanpath for `sentence_id=0` is the first review in the dump. Use it as a smell test: function words should be shorter and less fixated than `presents` / `failing` / `decency`.

## 6. Lock the fusion width before changing the head

```bash
python3 examples/scripts/06_fusion_shape_check.py
```

If I change `hidden_layer_size` in the training scripts, this script and `tests/test_fusion.py` should change in the same commit. The NumPy path is not trained; it only keeps the documented shapes honest.

## Importing from a notebook

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("examples").resolve()))

from gazekit.paths import default_paths
from gazekit.io import load_sentence_table
from gazekit.baselines import fit_gaze_only, fit_majority

df = load_sentence_table(default_paths().zuco_combined_standard)
print(fit_majority(df).as_dict())
print(fit_gaze_only(df).as_dict())
```

No need to spawn the scripts if you already have a REPL.

The numbers I got on this checkout are snapshotted in [example-results.md](example-results.md).
