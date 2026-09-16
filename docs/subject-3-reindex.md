# Subject 3 is packed, not merely short

This is the atlas’s most load-bearing CSV fact. If you average the twelve
`ZuCo_et_csv_data/{1..12}_SR.csv` files by row index, you are mixing different
movie reviews for every index ≥ 150.

## What the extractor does

`utils_ZuCo.DataTransformer.__call__` loops MATLAB `sentenceData` and
**continues** (skips) for Task 1, subject index 2 — which is CSV **subject 3**,
because `read_ZuCo_mat.py` writes `{i+1}_SR.csv`:

```python
if (self.task == 'task1' and subject == 2) and ((i >= 150 and i <= 249) or i == 399):
    continue
```

It also pre-allocates `len(data) - 101` feature rows for that subject. After
the loop it builds a DataFrame indexed `0 .. n-1` and `read_ZuCo_mat.py`
renames that index to `id`. The skipped original ids are **not** stored.

Packed length: 400 − 101 = **299**. Committed `3_SR.csv` has 299 rows, ids
0–298. That matches.

## The packing map

Original ids 0–149 are kept in place. Original 150–249 are dropped.
Original 250–398 slide down by 100 and occupy packed 150–298. Original 399
is dropped.

```
unpack(packed) = packed           if packed < 150
               = packed + 100     if packed ≥ 150

pack(150) = None
pack(250) = 150
unpack(150) = 250
unpack(298) = 398
```

`examples/sidecar/alignment.py` implements this. The tests require
`SentLen` to match subject 1 on the *unpacked* pairing for all 299 rows,
and they require the naive same-`id` pairing to fail at packed 150.

## Word-level smoking gun

Word tables use `Sent_ID` like `150_NR`. Subject 3 still *labels* that row
`150_NR`, but the tokens are original sentence 250:

| Reader | `Sent_ID` | first tokens |
| --- | --- | --- |
| subject 1 | `150_NR` | `its everything you dont go to the movies …` |
| subject 1 | `250_NR` | `the original wasnt a good movie but this …` |
| subject 3 | `150_NR` | `the original wasnt a good movie but this …` |

Tokens for original sentences 0–149 are identical (2594 rows). The first
DataFrame index where a positional average mixes reviews is **2594**.

Subject 3 also skipped original sentence 399 (9 word rows on subject 1),
which is why 7129 − 5293 = 1836 ≠ 1827 (the 150–249 block alone).
1827 + 9 = 1836.

## What `get_average_sentence_level.py` does

```python
average_df = pd.concat(dataframes).groupby(level=0).mean()
```

`pd.concat` keeps each frame’s `RangeIndex`, so row 0 of all twelve readers
is averaged (good — those are all original sentence 0), and row 150 of
subject 3 (original 250) is averaged with row 150 of everyone else (original
150). Rows 299–399 have eleven readers and no subject 3, which is a *missing
reader*, not a mixed sentence — slightly less wrong, still not “the mean
of twelve.”

The committed `average_data.csv` matches a recompute of that exact
`groupby(level=0)` (max abs diff on nFixations ~ 1e-15). The bug is in the
saved averages, not just in the script.

`ZuCo_et_csv_data/word/get_average.py` does the same thing at word level,
starting at row 2594.

## Prefix 0–149 is safe

`SentLen` matches subject 1 for packed ids 0–149 at rate 1.0. Word tokens
match. The hole starts at original sentence 150, which is MATLAB’s skip
block, not a truncated file.

## How to join subject 3 correctly

Do not join on `id`. Map packed → original with `unpack_subject3_id`, then
join to the 0–399 grid, leaving NaNs on 150–249 and 399. Average with
`skipna=True` on the original id. That is *not* what the 2024 scripts did.

`model_ZuCo_SST.py` never reads the per-subject files; it reads the already
joined `combined_sst_et_standard.csv`. If that join used the positional
averages, the 400-row training table inherits the mix for sentences ≥ 150.
