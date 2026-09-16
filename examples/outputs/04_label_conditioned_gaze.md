# Label-conditioned gaze (ZuCo standard table)

400 z-scored sentences. A mean of 0 is the corpus centre. Negative nFixations on the negative class means those sentences were, on average, fixated slightly less than the corpus mean — not that people closed their eyes.

| label | n | nFixations | FFD | GPT | TRT | GD | omissionRate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 negative | 123 | -0.0892 | -0.0784 | -0.0639 | -0.0960 | 0.0304 | 0.0351 |
| 1 neutral | 137 | 0.0831 | -0.0284 | 0.0429 | 0.0520 | -0.1387 | -0.1292 |
| 2 positive | 140 | -0.0029 | 0.0966 | 0.0141 | 0.0334 | 0.1090 | 0.0957 |

## Pearson r with integer label 0/1/2

| feature | r |
| --- | --- |
| nFixations | 0.0326 |
| FFD | 0.0715 |
| GPT | 0.0304 |
| TRT | 0.0509 |
| GD | 0.0356 |
| omissionRate | 0.0280 |

## Collinearity among fusion features

| a | b | r |
| --- | --- | --- |
| nFixations | TRT | 0.9581 |
| GPT | TRT | 0.9419 |
| nFixations | GPT | 0.9125 |
| TRT | GD | 0.7893 |
| nFixations | GD | 0.6973 |
| FFD | GD | 0.6794 |
| GPT | GD | 0.6746 |
| FFD | TRT | 0.6167 |
| FFD | GPT | 0.5470 |
| nFixations | FFD | 0.4213 |

## Why this matters for EyeTrackingModel

The fusion head sees nFixations, FFD, GPT, TRT, GD through one Linear(5 → 16). If TRT and GPT are highly correlated, that layer is not getting five independent cues. Label correlations here are small; gaze-only linear baselines in example 09 are the better floor for 'does gaze separate sentiment on its own?'.
