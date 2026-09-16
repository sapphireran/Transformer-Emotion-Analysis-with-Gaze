# Gaze features

The training scripts always feed **five** channels into
`EyeTrackingModel`. ZuCo tables carry three more that are useful for
analysis. Predicted SST tables only have the five.

## The five channels the models see

| ZuCo column | Full-SST column | Usual name | What it measures |
| --- | --- | --- | --- |
| `nFixations` | `nFix` | Number of fixations | How many times the eyes landed on the region |
| `FFD` | `FFD` | First fixation duration | Length of the very first fixation |
| `GD` | `GD` | Gaze duration (first-pass) | Sum of first-pass fixations before the eyes leave to the right |
| `TRT` | `TRT` | Total reading time | All fixations on the region, including later passes |
| `GPT` | `GPT` | Go-past / regression-path time | From first entering the region until it is passed to the right, including leftward regressions |

On **word** rows these are standard regions-of-interest measures (milliseconds
except `nFixations`). On **sentence** rows in this repo they are already
aggregates: `DataTransformer` sums word values and divides by the number of
words that had any fixation.

## Extra ZuCo channels

| Column | Meaning |
| --- | --- |
| `omissionRate` | Fraction of words in the sentence with no recorded fixation |
| `meanPupilSize` | Mean pupil size while reading the sentence (arbitrary camera units; raw mean ≈ 797) |
| `SFD` | Single-fixation duration — duration when the word received exactly one fixation |
| `SentLen` / `WordLen` | Token counts, not gaze |

`omissionRate` on the raw average table has mean **0.32** (min 0.16, max
0.60). Readers skip a lot of function words. That is expected in natural
reading and is why `nFixations` on a word can be 0.

## Typical raw magnitudes (subject-averaged sentences)

From `ZuCo_et_csv_data/average_data.csv`:

| column | mean | std | min | max |
| --- | ---: | ---: | ---: | ---: |
| SentLen | 17.81 | 8.06 | 3 | 42 |
| omissionRate | 0.32 | 0.07 | 0.16 | 0.60 |
| nFixations | 1.69 | 0.31 | 1.20 | 3.58 |
| meanPupilSize | 797 | 61 | 680 | 970 |
| GD (ms) | 141 | 21 | 108 | 273 |
| TRT (ms) | 203 | 48 | 131 | 427 |
| FFD (ms) | 117 | 8 | 102 | 166 |
| SFD (ms) | 72 | 11 | 39 | 122 |
| GPT (ms) | 242 | 57 | 153 | 587 |

First fixation around 100–130 ms and go-past times longer than gaze duration
is the usual reading-research pattern: people sometimes regress.

At the word level (`word_averages_v2.csv`) the distribution is wider because
zeros (skipped words) are kept. Mean `nFixations` drops to 1.11; max `GPT`
is 2425 ms (`decency` in sentence 0 is an extreme go-past).

## Collinearity

On the 400 standard-scaled ZuCo sentences, Pearson correlations among the
channels the model uses:

|  | nFixations | GD | TRT | FFD | GPT |
| --- | ---: | ---: | ---: | ---: | ---: |
| nFixations | 1.00 | 0.70 | **0.96** | 0.42 | **0.91** |
| GD | 0.70 | 1.00 | 0.79 | 0.68 | 0.68 |
| TRT | 0.96 | 0.79 | 1.00 | 0.62 | **0.94** |
| FFD | 0.42 | 0.68 | 0.62 | 1.00 | 0.55 |
| GPT | 0.91 | 0.68 | 0.94 | 0.55 | 1.00 |

`nFixations`, `TRT`, and `GPT` are almost the same sentence-level direction
after averaging 12 readers. Concatenating all five is still what the training
scripts do. A leaner probe would keep `FFD` (early) and `TRT` or `GPT` (late)
and drop the rest.

`SFD` moves the other way from `nFixations` (r = −0.59): sentences with many
fixations have fewer single-fixation words. `meanPupilSize` is nearly
uncorrelated with the fixation-time family (all |r| < 0.18).

## Sentiment is a weak linear companion

Mean standard-scaled gaze by SST-3 label (ZuCo join):

| label | omissionRate | nFixations | GD | TRT | FFD | GPT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 negative | 0.04 | −0.09 | 0.03 | −0.10 | −0.08 | −0.06 |
| 1 neutral | −0.13 | 0.08 | −0.14 | 0.05 | −0.03 | 0.04 |
| 2 positive | 0.10 | 0.00 | 0.11 | 0.03 | 0.10 | 0.01 |

The shifts are a few tenths of a standard deviation. Gaze-only linear models
should land near the 35% majority baseline. That is exactly what
`examples/06_text_vs_gaze_baselines.py` is for: if text-only already carries
the label, late fusion has to beat a strong bag-of-n-grams, not a dummy
classifier.

## Predicted vs human units

Full-SST sentence columns are z-scored predicted aggregates. Do not compare
`SST_data/combined_full_sst_et.csv` `TRT = -0.13` to ZuCo `TRT = 203 ms`.
The word-level predicted file (`prediction_test_v2.csv`) uses a different
scaling again (values typically in a 0–40 band, closer to the Provo export
than to milliseconds).

When you plot predicted and human gaze on the same axis, rescale first.
`examples/04_compare_scalers.py` only compares the three ZuCo sentence
tables, where the transform is a column-wise monotone map.

## Predicted SST channels collapse

On `SST_data/combined_full_sst_et.csv` the five predicted sentence
channels are almost the same direction:

| pair | Pearson r |
| --- | ---: |
| FFD–GPT | 0.999 |
| nFix–GPT | 0.998 |
| nFix–FFD | 0.995 |
| TRT–FFD | 0.995 |
| TRT–GPT | 0.995 |
| nFix–TRT | 0.986 |

Human ZuCo sentence averages are collinear but not like this (`FFD` vs
`nFixations` is only 0.42 there). The predictor has effectively produced
**one** reading-difficulty score copied into five columns. Concatenating
all five on Track B adds almost no extra information beyond a single
z-scored scalar.

