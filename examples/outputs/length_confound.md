# Length confound

On the **predicted / z-scored** full-SST sidecar, nFix, TRT, FFD, and GPT
all correlate around **r ≈ −0.30 to −0.37** with whitespace token count.
GD does not (near zero). Residualizing each feature against token count
barely changes r(feature, label), so the weak sentiment association is
not just 'longer reviews have a polarity'.

The negative length–gaze correlation is itself a property of the
*imputed* table: measured ZuCo gaze (per-word means aggregated to the
sentence) does not show the same strong anti-correlation. Treat the
full-SST sidecar as a model output, not as a millisecond stopwatch.

## Full SST train

n = 9482, mean tokens = 19.17, r(ntokens, label) = 0.0193

| feature | r_with_ntokens | r_with_label | r_residual_with_label |
| --- | --- | --- | --- |
| nFix | -0.3045 | -0.0663 | -0.0634 |
| GD | 0.0122 | -0.0701 | -0.0703 |
| TRT | -0.3720 | -0.0619 | -0.0589 |
| FFD | -0.3448 | -0.0652 | -0.0624 |
| GPT | -0.3326 | -0.0645 | -0.0616 |

## ZuCo-SST combined (standard scaled)

n = 400, mean tokens = 17.82, r(ntokens, label) = 0.0353

| feature | r_with_ntokens | r_with_label | r_residual_with_label |
| --- | --- | --- | --- |
| nFixations | -0.4675 | 0.0326 | 0.0556 |
| FFD | -0.4243 | 0.0715 | 0.0955 |
| GPT | -0.5378 | 0.0304 | 0.0586 |
| TRT | -0.5498 | 0.0509 | 0.0841 |
| GD | -0.4115 | 0.0356 | 0.0550 |
