# Subject-3 alignment audit

``utils_ZuCo.DataTransformer`` skips original sentences 150–249 and 399 for task-1 subject index 2 (file ``3_SR.csv``), then writes the remaining 299 rows with compacted ids 0..298. ``get_average_sentence_level.py`` averages the twelve CSVs by row index.

## Sanity checks

| check | value |
| --- | --- |
| first SentLen mismatch vs reader 1 | 150 |
| remapped SentLen matches reader 1 | yes |
| compact 150 maps to original | 250 |
| sentences with nFixations contamination | 246 |

## Compact id → original id

| compact id in 3_SR.csv | original sentence id |
| --- | --- |
| 0 | 0 |
| 149 | 149 |
| 150 | 250 |
| 298 | 398 |

## nFixations contamination by region

| region | ids | n | mean abs delta | max abs delta |
| --- | --- | --- | --- | --- |
| aligned_all_12 | 0–149 | 150 | 0.0000 | 0.0000 |
| contaminated_wrong_sentence | 150–249 | 100 | 0.0297 | 0.2111 |
| contaminated_and_shifted | 250–298 | 49 | 0.0381 | 0.1429 |
| reader3_missing_from_index | 299–398 | 100 | 0.0276 | 0.1717 |
| reader3_skipped | 399–399 | 1 | 0.0000 | 0.0000 |

## Worked ids

| id | SentLen index-mean | SentLen aligned | nFix index-mean | nFix aligned |
| --- | --- | --- | --- | --- |
| 0 | 22.0000 | 22.0000 | 2.1585 | 2.1585 |
| 149 | 25.0000 | 25.0000 | 1.5834 | 1.5834 |
| 150 | 9.5000 | 9.0000 | 1.6050 | 1.6236 |
| 250 | 14.3333 | 15.0000 | 1.6479 | 1.6812 |
| 298 | 14.1667 | 14.0000 | 1.6734 | 1.7859 |
| 398 | 16.0000 | 16.0000 | 2.2768 | 2.2037 |
| 399 | 9.0000 | 9.0000 | 1.4838 | 1.4838 |

## How to read this

Ids 0–149 are safe: all twelve readers still share the same sentence. From 150 onward, reader 3's compact id is a later sentence. The published ``average_data.csv`` follows the index mean, so it is slightly wrong on 246 sentences. The shift is small on nFixations (mean |delta| ≈ 0.03) because one reader out of twelve cannot move the mean far, but it is a real alignment bug, not noise.
