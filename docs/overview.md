# Project overview

## Research question

Does **human reading behavior** — or a model of it — help a transformer
decide whether a movie-review sentence is negative, neutral, or positive?

Affective language is not only a bag of polarity words. Readers slow down
on unexpected twists, skip function words, and regress when a clause
reverses sentiment ("…failing to provide a reason for us to care").
Eye-tracking corpora record those events as a handful of well-studied
timing features. This project asks whether concatenating those features
to a BERT or RoBERTa pooled representation improves three-class SST
accuracy over a text-only baseline.

## Why two tracks

Measured eye tracking does not exist for the full SST. ZuCo recorded
twelve readers on a **400-sentence** SST subset. That set is gold for
"does real gaze help?" but it is small, so `model_ZuCo_SST.py` uses
stratified 5-fold CV instead of a single holdout.

To train on the rest of SST, the repo stores **sentence-level gaze
columns already aligned** to all 11,853 sentences
(`SST_data/combined_full_sst_et.csv`). Those columns are standardized
(mean ≈ 0, std ≈ 1) and should be treated as **predicted or transferred**
gaze, not as twelve new human subjects. Word-level predictions for the
same sentences live in `gaze_prediction/data/prediction_test_v2.csv`
(191,971 tokens). `result/*.png` compares the shape of those predictions
to PROVO.

```
                 ┌── ZuCo 400 ── measured ET ── 5-fold CV ── model_ZuCo_SST.py
SST sentences ───┤
                 └── full 11.8k ── predicted ET ── 80/10/10 ── model_full_SST.py
```

## Architecture in one paragraph

`EyeTrackingModel` loads `bert-base-uncased` or `roberta-base`, runs the
sentence through the encoder, takes `pooler_output` (768-d), projects the
five gaze numbers through `nn.Linear(5, 16)`, concatenates to 784-d,
applies dropout 0.1, and classifies with `nn.Linear(784, 3)`. Text-only
modes swap the whole module for Hugging Face
`BertForSequenceClassification` / `RobertaForSequenceClassification`.
Tokenizer max length is 128. Loss is token-classification-style
cross-entropy on the three logits.

A numpy-only walkthrough of the same shapes is
`examples/06_feature_fusion_walkthrough.py`.

## What "emotion analysis" means here

The title says emotion; the labels are **SST polarity**: negative /
neutral / positive. There is no Ekman category head, no arousal/valence
regression, and no EEG branch even though ZuCo also recorded EEG. Pupil
size is stored on the ZuCo CSVs (`meanPupilSize`) and is a reasonable
arousal proxy, but it is not in the five-feature fusion vector.

## How to use this repo as a notebook

1. Read [datasets.md](datasets.md) and run `examples/01_inspect_datasets.py`.
2. Read [eye_tracking_features.md](eye_tracking_features.md) and run
   `examples/03_gaze_feature_stats.py` plus
   `examples/04_subject_variability.py`.
3. Read [models.md](models.md) and run
   `examples/06_feature_fusion_walkthrough.py`.
4. Read [known_issues.md](known_issues.md) **before** trusting a test-set
   number from `model_full_SST.py` (the test loop currently keeps only
   the last batch).
5. Only then install `requirements-train.txt` and train.

## Non-goals

- Not a production sentiment API.
- Not a re-implementation of the official SST or ZuCo loaders.
- Not company or client work.
- Social-media or product telemetry is out of scope.
