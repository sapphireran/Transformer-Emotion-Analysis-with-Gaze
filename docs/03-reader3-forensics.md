# Reader 3 forensics

This is the finding the rest of the lab book is built around.

## What the MATLAB helper actually dropped

`utils_ZuCo.DataTransformer.__call__` (task 1, subject 2) skips MATLAB
indices `150–249` and `399` — 101 sentences. The surviving 299 rows are
written to `ZuCo_et_csv_data/3_SR.csv` with a **new** `id` of 0…298.

```
compact 0..149  → original 0..149     (aligned)
compact 150..298 → original 250..398  (shifted by +100)
original 150..249 and 399 → absent
```

`gazebook.remap.compact_to_original(150)` is 250. Reader 1's sentence 150
is 9 words long; reader 3's compact row 150 is 15 words long. That 15-word
line is original sentence 250.

## How the published mean was built

`get_average_sentence_level.py` does **not** group by the `id` column in a
way that repairs the shift. It concatenates the twelve frames and takes
`groupby(level=0).mean()` after replacing 0 with NaN. That is a
**row-index** average.

`examples/02_reader3_forensics.py` rebuilds that positional 0→NaN mean and
matches `ZuCo_et_csv_data/average_data.csv` to ~1e-15. So the committed
averages are the contaminated ones, not a later cleaned export.

## What that does to sentence 150's *length*

`SentLen` is `len(sent.word)` — a property of the stimulus, not of the
reader. Eleven files store 9 at row 150. Reader 3 stores 15. The published
value is

```
(9 × 11 + 15) / 12 = 9.5
```

A half-word sentence length is impossible in the raw MATLAB. It is the
fingerprint of the index join.

`SentLen` disagrees with reader 1 on **147** of the 149 compact-shifted
rows (two accidental length ties). `nFixations` disagrees on **246**
sentences after a remapped 0→NaN mean is subtracted from the published
column. The worst nFixations shift is id **239** (about 0.21 fixations).

The three contaminated regions, if you remap reader 3 and keep the 0→NaN
rule:

| published id | what reader 3 contributed | what it should have contributed |
| --- | --- | --- |
| 0–149 | original 0–149 | original 0–149 (correct) |
| 150–249 | original 250–349 | nothing (those MATLAB rows were dropped) |
| 250–298 | original 350–398 | original 250–298 |
| 299–398 | nothing | original 299–398 |
| 399 | nothing | nothing (correctly dropped) |

## Word-level is worse

Sentence files at least *try* to be one row per sentence. Word files are a
long stream. Readers 1 and 3 share the same 2,594 tokens for sentences
0–149. At row 2,594, reader 1 stores sentence `150_NR` / `its` and reader
3 stores compact `150_NR` / `the` (original 250). From there on,
`word/get_average.py` — also a `groupby(level=0).mean()` — averages
**different words**. In the 5,293-row overlap there are **2,674**
reader-1 vs reader-3 word mismatches.

The first 150 sentences are the only span where a 12-reader mean is
semantically a 12-reader mean. Reliability numbers in
`docs/06-reader-reliability.md` are computed on that span only.

## Why this is not patched in `average_data.csv`

A later GPU run that still reads the committed averages should be able to
reproduce this clone. The personal layer reports the delta instead of
rewriting the 400-row means. Use `gazebook.remap.remapped_average` if you
want the repaired column in an example.
