# Shuffle-gaze control

Keep the sentiment labels. Permute gaze rows. Any *real* association
should disappear; whatever remains is sampling noise. This is the cheap
CPU version of the ablation you would want before claiming the 16-d
sidecar is doing causal work inside RoBERTa.

On full SST the real correlations are already |r| ≈ 0.06. After a
shuffle they sit around 0.00, as they should. The class-conditional
means also flatten. There is a weak pattern, but it is small enough
that a 784-d classifier can ignore it.

## Full SST train

Pearson r(feature, label), real vs row-shuffled gaze:

| feature | r_real | r_shuffled |
| --- | --- | --- |
| nFix | -0.0663 | 0.0190 |
| GD | -0.0701 | 0.0038 |
| TRT | -0.0619 | 0.0208 |
| FFD | -0.0652 | 0.0201 |
| GPT | -0.0645 | 0.0200 |

Class-conditional means (real):

| sentiment_label | nFix | GD | TRT | FFD | GPT |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0571 | 0.0597 | 0.0533 | 0.0559 | 0.0555 |
| 1 | 0.0982 | 0.1110 | 0.0913 | 0.0963 | 0.0968 |
| 2 | -0.0876 | -0.0921 | -0.0823 | -0.0867 | -0.0856 |

Class-conditional means (shuffled gaze, labels fixed):

| sentiment_label | nFix | GD | TRT | FFD | GPT |
| --- | --- | --- | --- | --- | --- |
| 0 | -0.0132 | -0.0042 | -0.0131 | -0.0138 | -0.0134 |
| 1 | -0.0092 | 0.0330 | -0.0212 | -0.0140 | -0.0128 |
| 2 | 0.0285 | 0.0044 | 0.0326 | 0.0303 | 0.0303 |

## ZuCo-SST combined

Pearson r(feature, label), real vs row-shuffled gaze:

| feature | r_real | r_shuffled |
| --- | --- | --- |
| nFixations | 0.0326 | -0.0024 |
| FFD | 0.0715 | -0.0332 |
| GPT | 0.0304 | -0.0231 |
| TRT | 0.0509 | -0.0107 |
| GD | 0.0356 | 0.0100 |

Class-conditional means (real):

| sentiment_label | nFixations | FFD | GPT | TRT | GD |
| --- | --- | --- | --- | --- | --- |
| 0 | -0.0892 | -0.0784 | -0.0639 | -0.0960 | 0.0304 |
| 1 | 0.0831 | -0.0284 | 0.0429 | 0.0520 | -0.1387 |
| 2 | -0.0029 | 0.0966 | 0.0141 | 0.0334 | 0.1090 |

Class-conditional means (shuffled gaze, labels fixed):

| sentiment_label | nFixations | FFD | GPT | TRT | GD |
| --- | --- | --- | --- | --- | --- |
| 0 | -0.0026 | 0.0263 | 0.0185 | 0.0016 | -0.0221 |
| 1 | 0.0103 | 0.0313 | 0.0212 | 0.0224 | 0.0159 |
| 2 | -0.0078 | -0.0538 | -0.0370 | -0.0233 | 0.0038 |
