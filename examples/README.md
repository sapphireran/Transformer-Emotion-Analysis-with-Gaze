# Examples

CPU-only walkthroughs on the committed CSVs. None of these scripts download
BERT / RoBERTa or write checkpoints.

Run every command from the **repository root** so relative data paths inside
the original training scripts stay valid if you mix the two styles. The
example modules resolve files from `examples/paths.py` and will still work
if you launch them via an absolute path.

```bash
python3 -m pip install -r requirements.txt
python3 examples/inspect_datasets.py
python3 examples/join_zuco_sst.py
python3 examples/subject_agreement.py
python3 examples/gaze_feature_stats.py
python3 examples/sentiment_baselines.py
python3 examples/fusion_toy.py
python3 examples/word_level_gaze.py
```

`--write path.json` dumps the numeric payload on the stats / baseline /
fusion / word-level scripts.

## What each script is for

| Script | Reads | Asks |
| --- | --- | --- |
| `inspect_datasets.py` | every major CSV | Are schemas, ids, and label sets intact? |
| `join_zuco_sst.py` | text + averaged gaze + combined tables | Does a fresh inner join match git? |
| `subject_agreement.py` | twelve `*_SR.csv` files | How noisy is the twelve-subject mean? |
| `gaze_feature_stats.py` | ZuCo + full SST combined tables | Do gaze columns differ by sentiment class? |
| `sentiment_baselines.py` | committed train/valid/test splits | Gaze-only vs length vs TF-IDF vs TF-IDF+gaze |
| `fusion_toy.py` | same splits | Linear concat analogue of `EyeTrackingModel` |
| `word_level_gaze.py` | word averages, predictions, PROVO | Are word tables finite and id-contiguous? |

`inspect_datasets.py` is the regression check. A non-zero exit means a
regenerate step dropped a class, a fusion column, or an id.

## Expected runtime

On this checkout, with the packages in `requirements.txt`:

- `inspect_datasets.py` — a few seconds (it loads `prediction_test_v2.csv`)
- `gaze_feature_stats.py` — under a second
- `sentiment_baselines.py` — about 10–30 seconds (full SST TF-IDF)
- `fusion_toy.py` — similar to the baselines
- `word_level_gaze.py` — a few seconds; word-id contiguity walks every sentence

## Reading the baseline table

Columns in the printed grid:

- `acc` — accuracy
- `f1_w` — weighted F1 (matches the training scripts)
- `f1_mac` — macro F1 (more honest on the small neutral class)
- `neg / neu / pos` — per-class F1

A useful personal result is **tfidf+gaze ≈ tfidf**. That does not mean the
transformer fusion is useless; it means any GPU gain has to come from
non-linear use of the same five numbers.

## Shared path map

`examples/paths.py` is the single list of file locations used by the
scripts. If you add a new derived table, register it there and extend
`inspect_datasets.py` rather than hardcoding a third copy of the path.
