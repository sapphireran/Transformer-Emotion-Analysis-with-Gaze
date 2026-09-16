# Overview

This repository studies **late fusion** of a transformer sentence embedding
and a small eye-tracking vector for **3-class movie-review sentiment**.

"Emotion analysis" in the project title is SST-3 polarity — negative, neutral,
positive — not a multi-emotion taxonomy (anger, joy, fear, …). The reviews are
the usual Rotten Tomatoes snippets that SST collected. The extra signal is
how people *read* those snippets.

## Why gaze would help

Eye-tracking measures are a noisy proxy for processing difficulty:

- More fixations and longer total reading time often mark unexpected words,
  rare names, or locally ambiguous syntax.
- Go-past time grows when the eyes regress to earlier material.
- Omission rate marks words the reader skipped entirely.
- Pupil size is a slower arousal / load signal.

Sentiment is not the same thing as processing difficulty. A savage one-liner
can be easy to read; a polite hedge can be hard. The bet in this repo is
weaker than "gaze predicts polarity." It is: **conditioned on the same
sentence, gaze might move the decision when the text embedding is unsure.**

That is why both training scripts keep a text-only baseline (`bert`,
`roberta`) next to a fused model (`bert_eye_tracking`, `roberta_eye_tracking`).
The fused model never replaces the transformer. It concatenates a 16-d
projection of the gaze vector onto the 768-d pooler output.

## Why two tracks

Human gaze for SST does not exist at full scale. ZuCo Task 1 recorded 12
readers on a few hundred movie reviews while they read normally (the NR /
"sentiment reading" condition). After cleanup this repo keeps **400** of
those sentences, each joined to an SST-3 label.

Four hundred rows is the honest human-gaze experiment. It is also too small
for a 110M-parameter encoder. The second track therefore **predicts** word
gaze for every SST sentence (11,853 rows) from a model that saw Provo and
ZuCo-style word tables, aggregates those words back to a sentence vector, and
trains the same fusion architecture on the large set.

```
              human gaze                         predicted gaze
                 │                                     │
                 ▼                                     ▼
        ZuCo Task 1 (NR)                      Provo + predictor
                 │                                     │
                 ▼                                     ▼
      400 sentences × 8 measures            11,853 sentences × 5 measures
                 │                                     │
                 ▼                                     ▼
         model_ZuCo_SST.py                      model_full_SST.py
           5-fold CV                              train/valid/test
```

The two tracks are not interchangeable ablations. A gain on full SST can come
from the predictor leaking text statistics into the "gaze" channels. A gain
on ZuCo is smaller-N but at least uses real eyes.

## What this docs/examples layer adds

The original repo was a handful of training and conversion scripts plus CSVs.
The docs now:

- name every checked-in table and the columns the models actually read
- define the reading measures in one place
- record how subject 3's missing sentences interact with index-based averaging
- ship examples that audit splits, plot word profiles, and fit linear
  text-vs-gaze baselines without a GPU

The original `model_*.py` files are left as the training entry points. The
examples library does not reimplement Hugging Face training.

## Personal scope

Work here is limited to this personal GitHub repository
(`sapphireran/Transformer-Emotion-Analysis-with-Gaze`). Derived numbers in
`docs/generated/` come from the CSVs in this clone, not from an external
dashboard or another project.
