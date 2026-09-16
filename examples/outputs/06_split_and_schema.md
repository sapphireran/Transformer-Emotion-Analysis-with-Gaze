# Split and schema audit

## ZuCo–SST (400)

| check | value |
| --- | --- |
| train / valid / test | 320 / 40 / 40 |
| combined | 400 |
| train∩valid | 0 |
| train∩test | 0 |
| valid∩test | 0 |
| missing from splits | 0 |
| extra in splits | 0 |
| combined labels 0/1/2 | 123 / 137 / 140 |
| train labels 0/1/2 | 103 / 107 / 110 |
| valid labels 0/1/2 | 7 / 14 / 19 |
| test labels 0/1/2 | 13 / 16 / 11 |

``model_ZuCo_SST.py`` ignores these 320/40/40 files and runs stratified 5-fold on the combined table. The 40-row valid split is not stratified (7 / 14 / 19). A 5-point accuracy swing there is two sentences.

## Full SST (11,853)

| check | value |
| --- | --- |
| train / valid / test | 9482 / 1185 / 1186 |
| combined | 11853 |
| train∩valid | 0 |
| train∩test | 0 |
| valid∩test | 0 |
| missing from splits | 0 |
| extra in splits | 0 |
| combined labels 0/1/2 | 4649 / 2241 / 4963 |
| train labels 0/1/2 | 3710 / 1833 / 3939 |
| valid labels 0/1/2 | 476 / 209 / 500 |
| test labels 0/1/2 | 463 / 199 / 524 |

These files *are* what ``model_full_SST.py`` trains on. Neutral is under-represented. See example 07 before quoting the printed test score.

## Normalized text overlap between tracks

| sentence | ZuCo label | SST label | agree |
| --- | --- | --- | --- |
| ... a roller-coaster ride of a movie | 2 | 2 | yes |
| girls gone wild and gone civil again | 1 | 1 | yes |
| under-rehearsed and lifeless | 0 | 0 | yes |

Only three strings match after light normalization. The 400-row ZuCo overlap is not a subset of ``combined_full_sst_et.csv`` in any useful sense — different tokenization and a different SST cut.
