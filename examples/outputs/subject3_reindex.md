# Subject-3 reindex (Task 1 hole)

`utils_ZuCo.DataTransformer` drops MATLAB subject 2 / CSV subject 3
sentences `150, 151, 152, 153, 154, ..., 249, 399` (101 rows) and then
writes a fresh contiguous `id`. The committed `3_SR.csv` therefore has
299 rows numbered 0..298, not a 400-row table
with holes. `SentLen` is a stimulus property, so it is a perfect key for
reconstructing the packing against subject 1.

- packed rows: 299 (expected 299)
- unpack map: packed `i<150` → original `i`; packed `i>=150` → original `i+100`
- example: packed 150 → original 250; packed 298 → original 398
- pack inverse: original 250 → packed 150; original 150 → None (skipped)

## Sentence-level SentLen pairing

Unpacked pairing matches SentLen on **299/299** rows.
Naively pairing the same `id` (what `groupby(level=0)` does) matches only **152/299** rows — exactly the prefix 0..149 plus whatever
later sentences happen to share a length by chance.

| packed_id | original_id | s3_SentLen | s1_SentLen | sentlen_match | same_row_would_match |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 22.0 | 22.0 | True | True |
| 149 | 149 | 25.0 | 25.0 | True | True |
| 150 | 250 | 15.0 | 15.0 | True | False |
| 151 | 251 | 19.0 | 19.0 | True | False |
| 298 | 398 | 16.0 | 16.0 | True | False |

## Word-level smoking gun

Subject-1 word table: 7129 tokens, 400 sentences.
Subject-3 word table: 5293 tokens, 299 sentences.
Tokens for original sentences 0-149: 2594 on both readers.
First mixed row if you average by DataFrame index: **2594**.

- subject 1, `Sent_ID=150_NR` tokens: `['its', 'everything', 'you', 'dont', 'go', 'to', 'the', 'movies']`
- subject 1, `Sent_ID=250_NR` tokens: `['the', 'original', 'wasnt', 'a', 'good', 'movie', 'but', 'this']`
- subject 3, `Sent_ID=150_NR` tokens: `['the', 'original', 'wasnt', 'a', 'good', 'movie', 'but', 'this']`

Subject 3's `150_NR` is the *text* of original sentence 250. Any word-level
`groupby(level=0).mean()` after row 2593 is averaging different reviews.

This is why `ZuCo_et_csv_data/average_data.csv` and `word/word_averages_v2.csv`
should not be treated as 'the mean reader on sentence k' for k ≥ 150.
