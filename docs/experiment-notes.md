# Experiment notes

Personal notes on why the defaults look the way they do. Not a paper.

## Why two datasets

ZuCo task 1 is the only place in this repo with **real** eye movements on **sentiment-labeled** sentences. Four hundred rows is enough to ask "does a 16-d gaze projection move CV accuracy at all?" and not enough to claim a general result.

SST is here so the same head can be trained where text is abundant and gaze is **synthetic**. If fusion helps on ZuCo and is flat on SST, the honest reading is that the transfer model is the bottleneck. If fusion helps on both, the concat head is doing something that survives noisy features.

## Why late concat, not cross-attention

The first version of this experiment was "can I ship a baseline this week", not "can I invent a multimodal transformer". Concat after pooling:

- keeps the Hugging Face encoder untouched,
- makes the text-only ablation a one-string change,
- and fails in an understandable way (the Linear can collapse to ~0).

Word-level gaze never enters the transformer. That is a limitation. A later personal experiment could broadcast the five features per token or build a 128 × 5 gaze channel. That is deliberately out of scope for the scripts that exist today.

## Why these five features

nFix, FFD, GPT, TRT, GD are the intersection of:

- columns present on both ZuCo sentence exports and the SST transferred tables,
- measures that show up in almost every reading-research baseline,
- and a width (5) that does not dwarf a 16-d projection.

SFD is collinear with FFD on single-fixation words. Pupil size is lighting-sensitive. Omission rate is interesting and unused — a cheap ablation is to add it as a sixth input.

## Why z-score on ZuCo

Raw TRT is hundreds of milliseconds; nFixations is O(1). Without scaling, the gaze Linear spends its first steps on magnitude, not pattern. Subject-averaged z-scoring across the 400 sentences is the simplest fix. Min–max is checked in as `combined_sst_et_min_max.csv` if you want a bounded variant; the training script does not read it.

## Why 5-fold on ZuCo and holdout on SST

400 rows → a single 40-row test file is a coin flip. Stratified 5-fold at least puts every sentence in a holdout once.

11,853 rows → holdout is fine, and `StratifiedKFold` × `roberta-base` × 5 epochs would be a long weekend. The existing 80/10/10 files are already there.

The ZuCo 320/40/40 CSVs exist because an earlier draft used holdout. CV replaced it in `model_ZuCo_SST.py`; the files stayed.

## Why weighted F1

SST neutrals are 2,241 / 11,853 ≈ 18.9%. Macro F1 would punish that class; weighted F1 tracks accuracy more closely and is what the scripts already print. When you compare runs, also write down **per-class recall for neutral**. That is where a gaze signal — if it exists — is most likely to show up (hedged, mixed, or sarcastic reviews).

## Why batch 256 on SST and 16 on ZuCo

SST + 128-token padding + 256 is a throughput choice for a single GPU. ZuCo with 16 is "small data, shuffle a lot". There is no sweep behind those numbers.

## Why RoBERTa as the default string

`roberta-base` was the stronger text-only starting point in informal personal runs on SST-style reviews. BERT remains in the switch for a matched pair. Do not compare `bert_eye_tracking` to `roberta` and attribute the gap to gaze.

## Things I would not do in a follow-up

- Train fusion for 20 epochs on 320 ZuCo rows without early stopping and then quote a single fold.
- Trust `model_full_SST.py` test printout until the `extend` bug is fixed.
- Average subject 3 with the others by row index.
- Call predicted SST gaze "eye tracking" in a figure caption.

## Things that are worth a follow-up

- Seed torch and the DataLoader generator; log `transformers.__version__`.
- Add macro F1 and a 3×3 confusion matrix to both scripts.
- Extract `EyeTrackingModel` so examples can import it without starting a run.
- Join word-level predicted gaze to SST sentences with an explicit reducer (mean, max TRT, etc.) checked into `examples/`.
- A gaze-shuffled control: same numbers, randomly reassigned to sentences. If fusion still "wins", the head is fitting noise.

`examples/dummy_baseline.py` already fits the gaze-only control. A shuffle control is the next honest check before another long SST train.
