# Sample results

First green `python3 examples/run_all.py` on this branch. Ridge numbers
are a teaching floor (hashed BOW + one-vs-rest), not RoBERTa.

## Contracts that the suite asserts

| measurement | value |
| --- | --- |
| ZuCo labels 0/1/2 | 123 / 137 / 140 |
| Full SST labels 0/1/2 | 4649 / 2241 / 4963 |
| Reader 3 sentence / word rows | 299 / 5,293 |
| nFixations sentences changed by remap | 246 (worst id 239, \|Δ\| ≈ 0.211) |
| SentLen sentences changed by remap | 147 |
| Published SentLen at id 150 | **9.5** = (9×11 + 15) / 12 |
| Word-stream first diverge row | 2,594 |
| Reader-1 vs reader-3 word mismatches | 2,674 / 5,293 overlap |
| Full-SST last test batch | 162 / 1,186 |
| Unused ZuCo valid mix | 7 / 14 / 19 |
| Standard scaler | population std (`ddof=0`) |

`average_data.csv` matches the positional 0→NaN mean to ~1e-15.
`combined_sst_et_standard.csv` matches `ssts_ZuCo.csv ⨝ standard_scaled_average_data.csv`.

## Gaze rank

| table | corr. condition number | smallest eigenvalue |
| --- | ---: | ---: |
| ZuCo recorded (400) | 304.4 | 0.0129 |
| Full SST projected (11,853) | 13,733 | 0.00033 |

Full-SST pairwise r among `nFix` / `FFD` / `GPT` / `TRT` ≥ 0.986.

## Reader reliability (sentences 0–149 only)

| feature | mean pairwise r | ICC(1) |
| --- | ---: | ---: |
| nFixations | 0.362 | 0.237 |
| TRT | 0.437 | 0.271 |
| GD | 0.355 | 0.237 |
| GPT | 0.332 | 0.211 |
| FFD | 0.102 | 0.052 |

## CPU ridge probes (5-fold, seed 42)

### ZuCo ∩ SST (400 recorded-gaze sentences)

| model | mean acc | acc sd | mean weighted F1 |
| --- | ---: | ---: | ---: |
| majority | 0.3500 | 0.0039 | 0.1815 |
| length only | 0.3351 | 0.0282 | 0.2383 |
| gaze only | 0.4104 | 0.0495 | 0.3895 |
| gaze ⟂ length | 0.4104 | 0.0483 | 0.4053 |
| hashed text (128-d) | 0.3575 | 0.0229 | 0.3541 |
| text + gaze | 0.3624 | 0.0246 | 0.3624 |
| text + shuffled gaze | 0.3801 | 0.0333 | 0.3786 |

Recorded gaze beats majority by about six points. A 128-d hash is a
worse linear text model than those five numbers on 400 short reviews;
concatenating the hash *hurts* the gaze-only score. Residualizing length
does not remove the gaze-only gain, so it is not just “short sentence.”

### Full SST (11,853 projected-gaze sentences)

| model | mean acc | acc sd | mean weighted F1 |
| --- | ---: | ---: | ---: |
| majority | 0.4187 | 0.0001 | 0.2472 |
| length only | 0.4197 | 0.0013 | 0.2576 |
| gaze only | 0.4231 | 0.0094 | 0.3623 |
| gaze ⟂ length | 0.4231 | 0.0097 | 0.3658 |
| hashed text (128-d) | 0.5037 | 0.0013 | 0.4519 |
| text + gaze | 0.5043 | 0.0036 | 0.4525 |
| text + shuffled gaze | 0.5038 | 0.0030 | 0.4520 |

Here the hash finally has enough rows to beat majority. Adding the
projected five-pack changes accuracy by +0.0006 — the same order as
shuffling gaze. That is what a rank-1 projected channel looks like.

## Word artifacts (spot checks)

- Sentence 4: text still says *empty*; every reader file stores `emp11111ty` (mean nFix 7.25, WordLen 10).
- Sentence 80: *murder-on-campus* → `murderoncampus` (mean nFix 6.67).
- 55 `unknown` placeholders keep `WordLen = 0`.
