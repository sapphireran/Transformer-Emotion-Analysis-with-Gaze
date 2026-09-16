# Gaze feature report

Integer labels: 0=negative, 1=neutral, 2=positive. Full-SST gaze columns are projected, not human recordings.

## ZuCo 5-d set used by the training scripts

Table `zuco_standard` (400 rows), feature set `zuco5`.

| feature    | mean neg | mean neu | mean pos | overall mean | std    |
|------------|----------|----------|----------|--------------|--------|
| nFixations | -0.0892  | 0.0831   | -0.0029  | -0.0000      | 1.0013 |
| FFD        | -0.0784  | -0.0284  | 0.0966   | -0.0000      | 1.0013 |
| GPT        | -0.0639  | 0.0429   | 0.0141   | 0.0000       | 1.0013 |
| TRT        | -0.0960  | 0.0520   | 0.0334   | 0.0000       | 1.0013 |
| GD         | 0.0304   | -0.1387  | 0.1090   | -0.0000      | 1.0013 |

Class-mean gap ranking (descriptive only):

| feature    | mean |class gap| |
|------------|------------------|
| GD         | 0.1651           |
| FFD        | 0.1167           |
| nFixations | 0.1149           |
| TRT        | 0.0987           |
| GPT        | 0.0712           |

Pearson(feature, integer label) — treat as a weak monotone check:

| feature    | r      |
|------------|--------|
| FFD        | 0.0715 |
| TRT        | 0.0509 |
| GD         | 0.0356 |
| nFixations | 0.0326 |
| GPT        | 0.0304 |

## ZuCo full sentence-level set

Table `zuco_standard` (400 rows), feature set `zuco_full`.

| feature       | mean neg | mean neu | mean pos | overall mean | std    |
|---------------|----------|----------|----------|--------------|--------|
| omissionRate  | 0.0351   | -0.1292  | 0.0957   | -0.0000      | 1.0013 |
| nFixations    | -0.0892  | 0.0831   | -0.0029  | -0.0000      | 1.0013 |
| meanPupilSize | -0.0021  | 0.1012   | -0.0972  | 0.0000       | 1.0013 |
| GD            | 0.0304   | -0.1387  | 0.1090   | -0.0000      | 1.0013 |
| TRT           | -0.0960  | 0.0520   | 0.0334   | 0.0000       | 1.0013 |
| FFD           | -0.0784  | -0.0284  | 0.0966   | -0.0000      | 1.0013 |
| SFD           | 0.0463   | -0.1878  | 0.1430   | 0.0000       | 1.0013 |
| GPT           | -0.0639  | 0.0429   | 0.0141   | 0.0000       | 1.0013 |

Class-mean gap ranking (descriptive only):

| feature       | mean |class gap| |
|---------------|------------------|
| SFD           | 0.2205           |
| GD            | 0.1651           |
| omissionRate  | 0.1499           |
| meanPupilSize | 0.1323           |
| FFD           | 0.1167           |
| nFixations    | 0.1149           |
| TRT           | 0.0987           |
| GPT           | 0.0712           |

Pearson(feature, integer label) — treat as a weak monotone check:

| feature       | r       |
|---------------|---------|
| FFD           | 0.0715  |
| TRT           | 0.0509  |
| SFD           | 0.0442  |
| meanPupilSize | -0.0412 |
| GD            | 0.0356  |
| nFixations    | 0.0326  |
| GPT           | 0.0304  |
| omissionRate  | 0.0280  |

## Full-SST projected 5-d set (train split)

Table `sst_train` (9482 rows), feature set `sst5`.

| feature | mean neg | mean neu | mean pos | overall mean | std    |
|---------|----------|----------|----------|--------------|--------|
| nFix    | 0.0571   | 0.0982   | -0.0876  | 0.0049       | 0.9895 |
| GD      | 0.0597   | 0.1110   | -0.0921  | 0.0066       | 0.9820 |
| TRT     | 0.0533   | 0.0913   | -0.0823  | 0.0043       | 0.9922 |
| FFD     | 0.0559   | 0.0963   | -0.0867  | 0.0045       | 0.9910 |
| GPT     | 0.0555   | 0.0968   | -0.0856  | 0.0049       | 0.9901 |

Class-mean gap ranking (descriptive only):

| feature | mean |class gap| |
|---------|------------------|
| GD      | 0.1354           |
| nFix    | 0.1239           |
| FFD     | 0.1220           |
| GPT     | 0.1216           |
| TRT     | 0.1157           |

Pearson(feature, integer label) — treat as a weak monotone check:

| feature | r       |
|---------|---------|
| GD      | -0.0701 |
| nFix    | -0.0663 |
| FFD     | -0.0652 |
| GPT     | -0.0645 |
| TRT     | -0.0619 |
