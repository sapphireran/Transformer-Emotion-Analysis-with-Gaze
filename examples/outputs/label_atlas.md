# Label atlas

Both experiment tracks use `sentiment_label ∈ {0, 1, 2}` for negative / neutral / positive.
The full-SST tables are z-scored gaze joined onto ~11.8k SST sentences.
The ZuCo-SST table is 400 actually-read reviews with *measured* gaze.

## Full SST — train

n = 9482

| label | name | n | share |
| --- | --- | --- | --- |
| 0 | negative | 3710 | 0.3913 |
| 1 | neutral | 1833 | 0.1933 |
| 2 | positive | 3939 | 0.4154 |

Majority baseline: always predict label 2 (positive) → accuracy 0.4154, weighted F1 0.2438.

## Full SST — valid

n = 1185

| label | name | n | share |
| --- | --- | --- | --- |
| 0 | negative | 476 | 0.4017 |
| 1 | neutral | 209 | 0.1764 |
| 2 | positive | 500 | 0.4219 |

Majority baseline: always predict label 2 (positive) → accuracy 0.4219, weighted F1 0.2504.

## Full SST — test

n = 1186

| label | name | n | share |
| --- | --- | --- | --- |
| 0 | negative | 463 | 0.3904 |
| 1 | neutral | 199 | 0.1678 |
| 2 | positive | 524 | 0.4418 |

Majority baseline: always predict label 2 (positive) → accuracy 0.4418, weighted F1 0.2708.

## Full SST — combined

n = 11853

| label | name | n | share |
| --- | --- | --- | --- |
| 0 | negative | 4649 | 0.3922 |
| 1 | neutral | 2241 | 0.1891 |
| 2 | positive | 4963 | 0.4187 |

Majority baseline: always predict label 2 (positive) → accuracy 0.4187, weighted F1 0.2472.

## ZuCo-SST combined (standard scaled)

n = 400

| label | name | n | share |
| --- | --- | --- | --- |
| 0 | negative | 123 | 0.3075 |
| 1 | neutral | 137 | 0.3425 |
| 2 | positive | 140 | 0.3500 |

Majority baseline: always predict label 2 (positive) → accuracy 0.3500, weighted F1 0.1815.

## ZuCo-SST train.csv (80/10/10, unused by model_ZuCo_SST.py)

n = 320

| label | name | n | share |
| --- | --- | --- | --- |
| 0 | negative | 103 | 0.3219 |
| 1 | neutral | 107 | 0.3344 |
| 2 | positive | 110 | 0.3438 |

Majority baseline: always predict label 2 (positive) → accuracy 0.3438, weighted F1 0.1759.

## ZuCo-SST valid.csv

n = 40

| label | name | n | share |
| --- | --- | --- | --- |
| 0 | negative | 7 | 0.1750 |
| 1 | neutral | 14 | 0.3500 |
| 2 | positive | 19 | 0.4750 |

Majority baseline: always predict label 2 (positive) → accuracy 0.4750, weighted F1 0.3059.

## ZuCo-SST test.csv

n = 40

| label | name | n | share |
| --- | --- | --- | --- |
| 0 | negative | 13 | 0.3250 |
| 1 | neutral | 16 | 0.4000 |
| 2 | positive | 11 | 0.2750 |

Majority baseline: always predict label 1 (neutral) → accuracy 0.4000, weighted F1 0.2286.

## Pearson r(feature, sentiment_label)

Weak associations on both tracks. Gaze is not a substitute for the text.

### Full SST train

| feature | r_with_label |
| --- | --- |
| nFix | -0.0663 |
| GD | -0.0701 |
| TRT | -0.0619 |
| FFD | -0.0652 |
| GPT | -0.0645 |

### ZuCo-SST combined

| feature | r_with_label |
| --- | --- |
| omissionRate | 0.0280 |
| nFixations | 0.0326 |
| meanPupilSize | -0.0412 |
| GD | 0.0356 |
| TRT | 0.0509 |
| FFD | 0.0715 |
| SFD | 0.0442 |
| GPT | 0.0304 |

## Class-conditional gaze means

### Full SST train

| sentiment_label | nFix | GD | TRT | FFD | GPT |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0571 | 0.0597 | 0.0533 | 0.0559 | 0.0555 |
| 1 | 0.0982 | 0.1110 | 0.0913 | 0.0963 | 0.0968 |
| 2 | -0.0876 | -0.0921 | -0.0823 | -0.0867 | -0.0856 |

### ZuCo-SST combined

| sentiment_label | omissionRate | nFixations | meanPupilSize | GD | TRT | FFD | SFD | GPT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0351 | -0.0892 | -0.0021 | 0.0304 | -0.0960 | -0.0784 | 0.0463 | -0.0639 |
| 1 | -0.1292 | 0.0831 | 0.1012 | -0.1387 | 0.0520 | -0.0284 | -0.1878 | 0.0429 |
| 2 | 0.0957 | -0.0029 | -0.0972 | 0.1090 | 0.0334 | 0.0966 | 0.1430 | 0.0141 |

On full SST, the **neutral** class sits at the *highest* mean nFix/GD/TRT,
and the **positive** class sits below zero (the z-scored mean). That is the
opposite of a simple 'harder text gets more fixations' story once the
sentences are already standardized.
