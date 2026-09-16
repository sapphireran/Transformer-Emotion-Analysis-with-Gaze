# Gaze features (personal glossary)

Eye-tracking features in this repo come from ZuCo's word-level MATLAB structs.
`utils_ZuCo.DataTransformer` copies each field if it is a scalar and writes `0`
if the field is missing or is an array.

I use these as **reading-process descriptors**, not as emotion labels. The
emotion / sentiment target is always the review label.

## The five features the trainers concatenate

| Code | Name | What a large value usually means |
| --- | --- | --- |
| `nFix` / `nFixations` | Number of fixations | The unit was looked at more often |
| `FFD` | First Fixation Duration | The first landing was long |
| `GD` | Gaze Duration (first pass) | First pass over the unit was long |
| `TRT` | Total Reading Time | Including look-backs, the unit soaked time |
| `GPT` | Go-Past Time | The reader struggled to leave the unit to the right |

On **word** rows these are milliseconds (or counts). On **sentence** rows
`DataTransformer` sums word values and divides by the number of fixated words,
so they are per-fixated-word means, not sentence totals.

## Extra ZuCo fields that are stored but not fused

| Code | Why it is interesting | Why the trainer ignores it |
| --- | --- | --- |
| `SFD` | Clean “one look” duration | Missing whenever a word was refixated |
| `meanPupilSize` | Arousal / load proxy, noisy | Different scale, lots of zeros for skips |
| `omissionRate` | How much of the sentence was skipped | Sentence-only; no word analogue in the fusion head |
| `SentLen` / `WordLen` | Length confound | Length is already in the text encoder |

## How sentence rows are built

From `utils_ZuCo.py`, sentence mode:

1. Walk each word in the sentence.
2. Add that word's `[nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT]`.
3. Count a word as fixated unless every one of those values is 0.
4. Divide the summed word features by `nwords_fixated`.
5. Store `SentLen = len(sent.word)` and `omissionRate` from the MATLAB sentence.
6. Drop rows that contain `+inf` / `-inf`.
7. Optionally min-max, mean-normalize, or z-score **per column**.

That last step is why `combined_sst_et_standard.csv` is safe to concat onto a
transformer: the five ET numbers are already on a roughly comparable scale.

## Word averages across 12 readers

`ZuCo_et_csv_data/word/get_average.py` concatenates the 12 word files and takes
the mean of the numeric ET columns **by row index**, then stitches identity
columns from subject 1. That only lines up if every subject file shares the
same token order. Subject 3 is shorter, so those later tokens are averaged
over fewer readers. `word_averages_v2.csv` fills leftover NaNs with 0 and
empty `Word` cells with `unknown`.

Of 7,129 averaged word rows, 91 have `nFixations == 0` (everyone skipped that
token, or the token was empty).

## Predicted gaze on full SST

`SST_data/convert_sst_to_et.py` does **not** look up ZuCo. It tokenizes each
SST sentence and writes zeros. A later gaze model (see
`gaze_prediction/data/`) fills those zeros. After pooling, the sentence table
`combined_full_sst_et.csv` has five z-scored numbers.

Those five predicted numbers are almost the same feature five times. On the
combined full SST table, Pearson correlations are:

|  | nFix | GD | TRT | FFD | GPT |
| --- | ---: | ---: | ---: | ---: | ---: |
| nFix | 1.00 | 0.78 | 0.99 | 1.00 | 1.00 |
| GD | 0.78 | 1.00 | 0.68 | 0.74 | 0.74 |
| TRT | 0.99 | 0.68 | 1.00 | 1.00 | 1.00 |
| FFD | 1.00 | 0.74 | 1.00 | 1.00 | 1.00 |
| GPT | 1.00 | 0.74 | 1.00 | 1.00 | 1.00 |

`GD` is the only predicted feature that is not a near-duplicate of `nFix`.
A linear layer of size 5→16 cannot invent independence that is not there.

Real ZuCo sentence features are correlated too (`nFixations`–`TRT` = 0.96,
`nFixations`–`GPT` = 0.91) but `SFD` moves the other way (`nFixations`–`SFD` =
−0.59), which is what you want if “many looks” and “single look duration”
are different stories.

## Correlation with sentiment (linear, not causal)

Computed on the checked-in combined tables:

**ZuCo standard (n=400)** — strongest linear link is still tiny:

- `FFD` +0.072
- `TRT` +0.051
- `SFD` +0.044
- `meanPupilSize` −0.041

**Full SST predicted (n=11,853)** — all five predicted features sit around
−0.05 to −0.06 with the label.

So gaze is a **weak linear companion** to sentiment in these files. If fusion
helps, it is because the transformer can use gaze as a small extra cue, not
because these five numbers already classify reviews.

Run `python examples/02_label_and_feature_stats.py` to regenerate the tables.
