# Feature dictionary

Personal glossary for the eye-tracking columns in this clone. Units in
the per-reader CSVs are close to milliseconds / counts / pupil-size
units. Units in the training tables are scaled.

## The five fusion inputs

`EyeTrackingModel` in both training scripts concatenates a 16-d
projection of these five numbers onto the 768-d pooler.

| name (ZuCo) | name (full SST) | what it is |
| --- | --- | --- |
| `nFixations` | `nFix` | Fixation count (word) or mean count over fixated words (sentence) |
| `FFD` | `FFD` | First Fixation Duration |
| `GPT` | `GPT` | Go-Past Time (includes regressions before leaving to the right) |
| `TRT` | `TRT` | Total Reading Time (all passes) |
| `GD` | `GD` | Gaze Duration / first-pass time |

On the 400-row **z-scored** ZuCo table these five are badly
collinear. From `examples/04_label_conditioned_gaze.py`:

| pair | Pearson r |
| --- | ---: |
| nFixations–TRT | 0.958 |
| GPT–TRT | 0.942 |
| nFixations–GPT | 0.913 |
| TRT–GD | 0.789 |

A `Linear(5 → 16)` is not seeing five independent cues. nFixations and
TRT are almost the same direction.

## Present on ZuCo, not fused

| name | what it is |
| --- | --- |
| `SFD` | Single Fixation Duration (FFD when there was exactly one fixation) |
| `omissionRate` | Fraction of words with no fixation |
| `meanPupilSize` | Mean pupil size over fixations |
| `SentLen` | Token count in the ZuCo sentence struct |
| `WordLen` | Character length of the cleaned word |

`SentLen` is the cheapest alignment key in the clone. Reader 3's
`SentLen` matches reader 1 on ids 0–149 and then jumps — that is how
the remapping was found.

## Scaling

| table | typical `nFixations` / `nFix` | typical pupil |
| --- | --- | --- |
| `ZuCo_et_csv_data/1_SR.csv` | ~1–3 | ~800–1000 |
| `ZuCo_et_csv_data/average_data.csv` | ~2.16 on sentence 0 | ~851 |
| `combined_sst_et_standard.csv` | z-score, mean 0, std 1 | z-score |
| `combined_sst_et_min_max.csv` | 0–1 | 0–1 |
| `SST_data/combined_full_sst_et.csv` | z-score (projected) | (column absent) |
| `gaze_prediction/data/provo.csv` | ~15 | (column absent) |

`model_ZuCo_SST.py` reads the **standard** table. Mixing a raw pupil
column into that tensor will dominate the gaze layer.

`get_average_sentence_level.py` replaces zeros with NaN before the
mean, then min-max and z-score writes. A simple arithmetic mean of the
twelve files is therefore not identical to `average_data.csv` on every
row (max \|nFixations\| gap on the order of 0.19).

## Label-conditioned means (z-scored ZuCo)

| label | n | nFixations | TRT | omissionRate |
| --- | ---: | ---: | ---: | ---: |
| negative | 123 | −0.089 | −0.096 | +0.035 |
| neutral | 137 | +0.083 | +0.052 | −0.129 |
| positive | 140 | −0.003 | +0.033 | +0.096 |

Pearson r with the integer label 0/1/2 is under 0.08 for every fused
feature. Gaze is not a surrogate sentiment label on this table. See
[hypotheses](hypotheses.md) and example 09.

## Sentence vs word construction

`DataTransformer` (sentence mode) sums word measures and divides by
the number of **fixated** words, not by `SentLen`. Skipped words do
not pull the sentence average toward zero. The word table keeps those
zeros. Example 05 walks sentence 0: mean word `nFixations` is 1.61,
the published sentence mean is 2.16.
