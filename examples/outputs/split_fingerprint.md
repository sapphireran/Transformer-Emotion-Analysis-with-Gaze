# Split fingerprint

## Full SST (`SST_data/spilt.py`, seed 42, 80/10/10)

train 9482 + valid 1185 + test 1186 = 11853 (combined 11853).
Ids partition the combined table. Duplicate *texts* still leak across splits
because `train_test_split` was keyed on rows, not unique strings.

| pair | id_overlap | text_overlap |
| --- | --- | --- |
| train ∩ valid | 0 | 1 |
| train ∩ test | 0 | 1 |
| valid ∩ test | 0 | 0 |

Duplicate sentence strings inside train: 11 extra rows (9471 unique / 9482 rows).
Combined table has 13 extra duplicate-text rows (11840 unique / 11853).

Train/valid shared text:

- `For a film that 's being advertised as a comedy , Sweet Home Alabama is n't as funny as you 'd hoped .`

Train/test shared text:

- `` Stock up on silver bullets for director Neil Marshall 's intense freight train of a film . '`

## ZuCo-SST 80/10/10 CSVs vs what the trainer actually does

`ZuCo_SST_data/{train,valid,test}.csv` is 320/40/40.
`model_ZuCo_SST.py` **ignores those files** and runs `StratifiedKFold(5)` on
`combined_sst_et_standard.csv` instead. The 40-row valid split is also
badly imbalanced (only 7 negatives), which is one reason a stratified
k-fold on all 400 rows is the more defensible protocol — if you remember
that those CSVs are leftover from an earlier split script (`spilt.py`).

| pair | id_overlap | text_overlap |
| --- | --- | --- |
| train ∩ valid | 0 | 0 |
| train ∩ test | 0 | 0 |
| valid ∩ test | 0 | 0 |

## Two almost-disjoint corpora

Only **3** review strings appear in both the 400-row ZuCo
table and the 11.8k full-SST table. Measured gaze and predicted gaze are
not two views of the same items.

- `... a roller-coaster ride of a movie`
- `Girls gone wild and gone civil again`
- `under-rehearsed and lifeless`
