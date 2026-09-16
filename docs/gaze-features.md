# Gaze feature dictionary

Eye-tracking columns in this repo use the names from ZuCo's word structs and
the sentence-level aggregation in `utils_ZuCo.DataTransformer`. Durations are
milliseconds in the **raw** subject files. After min-max or standard scaling
they are unitless.

## Features the fusion models actually use

`EyeTrackingModel` always reads **five** sentence-level channels.

| ZuCo-SST column | Full-SST column | Psycholinguistic name | What a large value usually means |
| --- | --- | --- | --- |
| `nFixations` | `nFix` | Number of fixations | The sentence attracted more looks (or more refixations) |
| `FFD` | `FFD` | First Fixation Duration | The first landing was long — early lexical difficulty |
| `GPT` | `GPT` | Go-Past Time (regression-path) | The reader went back left before leaving the region |
| `TRT` | `TRT` | Total Reading Time | All looks, including later rereading, summed |
| `GD` | `GD` | Gaze Duration (first-pass) | Time in the region before the eyes first left it |

In the original scripts these five numbers are a **linear** projection to 16
dimensions. There is no per-word attention over gaze, and no attempt to align
fixations to BPE tokens.

## Extra columns stored but not fused

| Column | Level | Meaning |
| --- | --- | --- |
| `SFD` | word / sentence | Single Fixation Duration — the look when a word is fixated exactly once |
| `meanPupilSize` | word / sentence | Mean pupil size during reported fixations (arousal / luminance confound) |
| `omissionRate` | sentence | Fraction of words with no fixation (`sent.omissionRate` from ZuCo) |
| `SentLen` | sentence | Number of words in the ZuCo sentence object |
| `WordLen` | word | Character length of the cleaned token |
| `Sent_ID` | word | `{sentence_index}_NR` for Task 1/2, `{sentence_index}_TSR` for Task 3 |
| `Word_ID` | word | 0-based position inside the sentence |
| `Word` | word | Token after stripping punctuation; sentence-initial words lowercased |

`model_ZuCo_SST.py` could have used SFD, pupil, or omission rate. It does not.
If you add them, bump `num_eye_tracking_features` and the CSV column list
together.

## How sentence-level values are built

For each sentence, `DataTransformer` walks `sent.word`:

1. Read the raw word attributes (`nFixations`, `meanPupilSize`, `GD`, `TRT`,
   `FFD`, `SFD`, `GPT`). Missing or `ndarray` attributes become 0.
2. Sum those word vectors into the sentence.
3. Divide by **the number of words that had any non-zero feature**, not by
   `SentLen`. Skipped words therefore do not dilute the average.
4. Store `SentLen = len(sent.word)` and `omissionRate` from the sentence
   object itself.

That is why a five-word sentence with two skipped words can still have a
high mean TRT: the mean is over the three fixated words.

## Missing data and zeros

ZuCo marks unfixated words with zeros. The averaging script
`get_average_sentence_level.py` turns those zeros into NaN **before** taking
the cross-subject mean, then writes two scaled tables:

- `min_max_scaled_average_data.csv`
- `standard_scaled_average_data.csv`

`DataTransformer` can instead fill NaNs with zeros, the column mean, or the
column min (`fillna='zeros'|'mean'|'min'`).

The helper implementations live in `gaze_emotion.scaling`.

## Scaling families

| Name | Formula (per column) | Used by |
| --- | --- | --- |
| `raw` | unchanged | `read_ZuCo_mat.py` (per-subject dumps) |
| `min-max` | `(x - min) / (max - min)` | `DataTransformer`, sklearn path |
| `mean-norm` | `(x - mean) / (max - min)` | `DataTransformer` only |
| `standard` | `(x - mean) / std` | `DataTransformer`, sklearn path, **ZuCo training default** |

Full-SST projected features are already scaled in
`SST_data/combined_full_sst_et.csv`; the training script does not scale them
again.

## Word-level predicted gaze

`gaze_prediction/data/prediction_test_v2.csv` and `provo.csv` are
**word-level** tables (`sentence_id`, `word_id`, `word`, plus gaze). PROVO
uses `fixProp` instead of `GD`. These files feed the projection step that
produced full-SST sentence vectors; the transformer training scripts never
open them directly.

## Reading a single example

Take sentence 3 in `ZuCo_SST_data/combined_sst_et_standard.csv`:

> "Slow, silly and unintentionally hilarious."

It is labeled **neutral** (`1`) and has an unusually large standardized
`nFixations` / `TRT` / `GPT`. Short, punchy reviews can still demand extra
looks when the wording is odd — which is exactly the kind of signal late
fusion is supposed to give the classifier. `examples/gaze_feature_tour.py`
prints several such rows next to their labels.
