# Reader reliability (clean 150 only)

An ICC that silently includes reader 3's compact ids 150–298 is an ICC
on mismatched stimuli. The numbers below use original sentences **0–149**
only — the span where all twelve `*_SR.csv` files still refer to the same
review.

## Pairwise Pearson r and ICC(1)

From `examples/05_reader_reliability.py` on this clone:

| feature | mean pairwise r | min r | max r | ICC(1) |
| --- | ---: | ---: | ---: | ---: |
| nFixations | 0.362 | 0.101 | 0.602 | 0.237 |
| TRT | 0.437 | 0.223 | 0.704 | 0.271 |
| GPT | 0.332 | 0.075 | 0.560 | 0.211 |
| GD | 0.355 | −0.069 | 0.649 | 0.237 |
| FFD | 0.102 | −0.202 | 0.440 | 0.052 |
| omissionRate | 0.284 | −0.003 | 0.506 | 0.167 |

TRT is the most agreed-upon duration. FFD is barely above noise between
some reader pairs (negative mins). nFixations ICC(1) ≈ 0.24 is *fair /
poor* in the usual psychometrics bins: a 12-reader mean is doing real
work, and a single reader would be a bad stand-in.

## Why averaging still makes sense — and why the index bug hurts

A noisy but positive ICC is exactly why the repo averages readers before
training. The positional join after sentence 149 folds the wrong reader's
sentence into 246 published means. That does not destroy the table — the
typical |Δ| on nFixations is ~0.02 — but it is a systematic error, not
measurement noise, and it is concentrated on ids 150–398.

## Do not compute ICC on the word-average file after row 2594

Those rows are not twelve observations of the same word.
