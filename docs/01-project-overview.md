# Project overview

## Why combine gaze with a transformer?

A sentiment classifier that only sees tokens has to infer affect from
lexical and syntactic cues. Human readers also leave a **behavioral
trace**: they skip words, linger on unexpected adjectives, or regress
when a twist appears late in the sentence.

Eye-tracking records that trace. If those signals are even partly
independent of the bag of words, adding them to `pooler_output` should
help on borderline reviews — the ones that mix praise and complaint, or
that use irony.

This repo tests that idea in the narrow setting of **movie-review
sentences** that (a) have SST-style 3-class labels and (b) either were
read under ZuCo Task 1, or have **estimated** gaze columns so the same
fusion head can run at SST scale.

## What is being classified?

A single English sentence → `{0: negative, 1: neutral, 2: positive}`.

There is no token-level emotion tagger, no EEG branch (ZuCo recorded
EEG, but `EyeTrackingModel` never loads it), and no subject-identity
input. Twelve ZuCo readers are averaged **before** training so the
classifier never sees who was reading.

## Claims this code can and cannot support

**Can support (if you train and log carefully):**

- Does a linear gaze projection concatenated onto BERT/RoBERTa beat the
  same encoder without gaze, on the 400-sentence measured set (5-fold)?
- Does the same head still help when gaze is transferred onto ~12k SST
  sentences (fixed split)?
- How sensitive is that gap to scaling (`standard` vs `min-max`) and to
  which five features you keep?

**Cannot support from this checkout alone:**

- Published accuracy numbers — there is no `result/` metrics table, only
  exploratory plots.
- Causal claims (“readers linger because the sentence is negative”).
- Cross-domain emotion (tweets, stories, dialogue). SST + ZuCo Task 1 is
  movie-review language.
- Word-level fusion. Both training scripts collapse the sentence to one
  5-d gaze vector plus one pooled text vector.

## Design choices that matter

1. **Late fusion, not cross-attention.** Gaze never attends over tokens.
   A 5-d vector is projected to 16-d and glued to the CLS/pooler state.
   That is cheap and easy to ablate, but it cannot say *which* word was
   fixated.
2. **Subject-averaged gaze.** Inter-reader variance is treated as noise.
   That matches a “generic reading difficulty” story and mismatches a
   “this reader is confused / delighted” story.
3. **Two scales of supervision.** 400 gold sentences vs 11k sentences
   with synthetic gaze. Improvements on the large set can be an artifact
   of the gaze **estimator**, not of real oculomotor signal.
4. **Hard-coded hyperparameters** at the top of each training script.
   There is no CLI. Changing `model_type` is an edit-and-rerun workflow.

## How the pieces connect

```
ZuCo .mat (not in repo)
        │
        ▼
utils_ZuCo.DataTransformer  ──►  ZuCo_et_csv_data/{1-12}_SR.csv
        │
        ▼
get_average_sentence_level  ──►  average + min-max + standard CSVs
        │
        ▼
join with ssts_ZuCo.csv     ──►  ZuCo_SST_data/combined_sst_et_*.csv
        │
        ├── model_ZuCo_SST.py          (measured, CV)
        │
        └── word-level averages
                 │
                 ▼
         gaze_prediction/ convert     (scale into 0–100-ish word rows)
                 │
                 ▼
         SST_data/*_full_sst.csv
                 │
                 ▼
         model_full_SST.py            (transferred, fixed split)
```

The `examples/` scripts only read the derived CSVs. They are the
fast path for checking schemas, class balance, and a gaze-only
baseline before you spend a GPU night on RoBERTa.

## Related personal notes

- Feature definitions: [03-eye-tracking-features.md](03-eye-tracking-features.md)
- Exact fusion math: [05-model-architecture.md](05-model-architecture.md)
- Implementation traps: [08-known-quirks.md](08-known-quirks.md)
