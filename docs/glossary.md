# Glossary

Short definitions as used in this personal repo. Reading-research
terms follow the usual eye-tracking sense; dataset names follow the
papers cited in the root README.

**BERT / RoBERTa.** Transformer encoders. Here: `bert-base-uncased` and
`roberta-base`, 768-d hidden size, Hub checkpoints.

**Early / late fusion.** Early: gaze enters the encoder (per-token
embeddings or cross-attention). Late: gaze meets text only after both
have been reduced to vectors. These trainers are late fusion.

**FFD (first fixation duration).** Length of the first fixation on a
word.

**Fixation.** Period the eye stays relatively still on a landing
position. Contrasts with saccades (the jumps).

**GD (gaze duration / first-pass time).** Sum of first-pass fixations
on a word before the eyes leave it to the right.

**Go-past time (GPT).** Time from first entering a word until passing
it to the right, including regressions to earlier words.

**Late fusion head.** `Linear(5 → 16)` on gaze, concat with pooler,
`Dropout(0.1)`, `Linear(784 → 3)`.

**Macro F1.** Unweighted mean of per-class F1. Treats neutral as equal
to pos/neg. Not computed in the historical trainers.

**nFix / nFixations.** Fixation count. Sentence-level ZuCo values are
already divided by the number of fixated words.

**NR / TSR.** ZuCo task tags on `Sent_ID`: normal reading vs
task-specific reading. Task 1 files here use `_NR`.

**Omission / skip.** A word with no fixation. Stored as an all-zero
gaze row at word level; summarised as `omissionRate` at sentence level.

**Pooler output.** Sentence vector from the encoder (`pooler_output`),
not a mean of word-pieces.

**Predicted / projected gaze.** Model-emitted reading-time features on
full SST. Not measured on those sentences.

**PROVO.** Provo Corpus of eye movements. `gaze_prediction/data/provo.csv`
is a small extract; `fixProp` is fixation probability, not GD.

**Regression.** Saccade back to an earlier word. Inflates GPT and
often TRT relative to GD.

**SFD (single fixation duration).** Duration when a word was fixated
exactly once. Often zero on re-read words.

**SR.** Sentiment reading. Filename tag on the 12 subject CSVs.

**SST.** Stanford Sentiment Treebank. Ternary labels in this clone
(the original also has fine-grained 0–1 scores and a binary subset).

**StratifiedKFold.** Class-balanced folds. Used on the 400-sentence
set with `n_splits=5`, `random_state=42`.

**TRT (total reading time).** Sum of all fixation durations on a word,
including later visits.

**Weighted F1.** F1 averaged with class support as weights. Close to
accuracy when classes are mildly imbalanced.

**ZuCo.** Zurich Cognitive Language Processing Corpus: simultaneous
EEG + eye-tracking during reading. This clone keeps ET only, Task 1,
12 subjects, 400 SST movie-review sentences.

**ZuCo ∩ SST.** The 400-sentence overlap used for measured-gaze
experiments.
