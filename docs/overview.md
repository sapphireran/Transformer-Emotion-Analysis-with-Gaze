# Research overview

## Question

Does a sentence-level eye-tracking vector improve transformer sentiment classification?

The text-only baseline is a fine-tuned BERT or RoBERTa classifier. The gaze-aware variant keeps the same encoder and adds five reading-time numbers that were either:

- **recorded** while people read the sentence (ZuCo Task 1), or
- **predicted** from word-level gaze models trained on other reading corpora (full SST).

If gaze is only a noisy restatement of length or word frequency, the extra branch should not help. If it encodes processing difficulty that correlates with polarity or sarcasm, the fused model should win on accuracy / macro-ish weighted F1.

## Why gaze might matter for emotion

Sentiment labels are about the *writer’s* stance. Gaze is about the *reader’s* effort.

Those are not the same thing, but they overlap in practice:

- Negation, sarcasm, and garden-path structures slow first-pass reading (FFD, GD) and inflate go-past time (GPT).
- Highly charged adjectives often attract extra fixations (`nFixations`) and longer total reading time (TRT).
- Omission rate and single-fixation duration (SFD) mark words that were skipped or resolved in one glance.

A linear 5 → 16 projection is a deliberately weak gaze encoder. The experiment is “does *any* of this vector help?”, not “what is the best gaze architecture?”. Word-level cross-attention or a gaze-conditioned adapter would be a later personal follow-up, not what these scripts implement.

## Design choices that are intentional

**Late fusion, not early fusion.** The transformer never sees gaze tokens. Gaze is concatenated after pooling. That keeps the Hugging Face checkpoints intact and makes the ablation (`bert` vs `bert_eye_tracking`) a clean on/off switch.

**Five features, not the full ZuCo set.** Sentence-level ZuCo CSVs also have `SentLen`, `omissionRate`, `meanPupilSize`, and `SFD`. The training scripts drop those and keep `{nFixations, FFD, GPT, TRT, GD}`. Length is already implicit in the token sequence. Pupil size is more arousal than polarity. SFD is sparse at sentence average. The five-feature slice is the one both `model_*.py` files actually consume.

**Two corpus sizes.** 400 real-gaze sentences are the honest test of the hypothesis but are too small for a 110M encoder. 11.8k SST sentences are large enough to fine-tune RoBERTa but the gaze column is predicted. Reporting both avoids pretending one setting answers both validity and scale.

**Subject-averaged gaze.** ZuCo has 12 readers. The joined tables use the mean across subjects after treating zeros as missing. That throws away inter-reader variance (which is scientifically interesting) in exchange for one vector per sentence.

## What this repo is not

- It is not a production sentiment API.
- It is not an EEG project. ZuCo has EEG channels; `utils_ZuCo.py` only lifts eye-tracking fields.
- It is not a gaze *prediction* paper. `gaze_prediction/data/` stores inputs and outputs of that side experiment; the predictor weights are not in this snapshot.
- It is not company or client work. Paths, comments, and notes are personal lab notes.

## Mental model of a forward pass

```
sentence ──► tokenizer (max_length=128) ──► BERT/RoBERTa ──► pooler_output (768)
                                                                    │
nFix, FFD, GPT, TRT, GD ──► Linear(5, 16) ──────────────────────────┤
                                                                    ▼
                                                         concat → Dropout(0.1)
                                                                    ▼
                                                         Linear(784, 3) → logits
```

Text-only mode skips the middle branch and uses the stock classification head.

## Success criteria (personal)

1. On ZuCo 5-fold CV, `*_eye_tracking` should not *lose* badly to text-only if gaze is well scaled. A small gain is enough to keep the line of work.
2. On full SST, a gain is only meaningful if the predicted gaze is not a leak of the label (for example via a predictor that saw sentiment). See `docs/experiments.md`.
3. The data join must be sentence-aligned. `examples/sentence_gaze_join.py` checks that the 400 ZuCo texts and the 400 averaged gaze rows share ids.

## Reading order

If you only want to understand the code, read in this order:

1. This page
2. `docs/datasets.md` — what each CSV is
3. `docs/model-architecture.md` — the Python classes
4. `docs/data-pipeline.md` — how the CSVs were built
5. `examples/` — run the inspections before touching a GPU
