# What the example scripts are supposed to show

This is a checklist I use after changing `tea_gaze` or a CSV. Fresh numbers
live in the script output; the notes below are the invariants.

## `01_inspect_datasets.py`

- ZuCo combined tables: 400 rows.
- Full SST combined: 11,853 rows; train 9,482; valid 1,185; test 1,186.
- Subject 3 sentence file: 299 rows. Everyone else: 400.
- `word_averages_v2.csv`: 7,129 rows, 400 distinct `Sent_ID`.

## `02_label_and_feature_stats.py`

- ZuCo labels 123 / 137 / 140.
- Full SST labels 4,649 / 2,241 / 4,963.
- Absolute gaze–label correlations stay under 0.10.
- Full SST lists several |r| ≥ 0.95 pairs among `nFix`, `FFD`, `TRT`, `GPT`.

## `03_compare_scaling.py`

- Rebuilt standard columns have mean ≈ 0 and std ≈ 1.
- Rebuilt min-max columns sit in [0, 1].
- `max |rebuilt - committed|` should be tiny (1e-6 or better) if
  `average_data.csv` is still the parent of the two scaled files.
  Last run on this laptop: standard `2.2e-15`, min-max `5.6e-16`.

## `04_toy_fusion_forward.py`

- Prints concat size 64 for the toy geometry (48 + 16).
- Prints concat size 784 for the real geometry (768 + 16).
- Holdout table has three rows; none of them is a GPU result.

## `05_gaze_only_baseline.py`

- Five folds print for each corpus.
- Gaze-only logistic is allowed to lose to majority. That is a finding,
  not a failure of the script.

## `06_word_to_sentence.py`

- Rebuilt table has 400 sentence ids.
- `n_words` sums to 7,129.
- Mean absolute diffs vs `average_data.csv` can be large. The script says
  why.

## `07_schema_and_glossary.py`

- Four `model_type` strings.
- Unused ZuCo columns: `omissionRate`, `meanPupilSize`, `SFD`.

## `08_split_sanity.py`

- All three pairwise id intersections are 0.
- Union of split ids equals the combined table.
- Prints the reminder that only the full-SST trainer consumes its split
  files.
