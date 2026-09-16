# Design notes

Personal reasoning behind the choices that are already in the trainers
and the ones I would make next. Not a paper; not a company design doc.

---

## Why late fusion, not token-level gaze

The measured word table exists (`ZuCo_et_csv_data/word/`, 7,129 rows).
A cleaner linguistic model would align those rows to RoBERTa word-pieces
and add a gaze embedding per piece.

That alignment is annoying:

- ZuCo `Word` is punctuation-stripped and first-token lowercased.
- RoBERTa uses a byte-level BPE; `presents` is not guaranteed to be one
  piece.
- Skipped words have zeros; those zeros must not look like “very short
  fixations” after LayerNorm unless you also pass a skip bit.
- Full SST only has *predicted* word gaze, so any token-level model
  inherits the predictor’s errors at the worst possible place (inside
  the encoder).

Late fusion sidesteps all of that. It also matches a weak hypothesis:
**sentence-level reading effort** is a small extra signal for a 3-way
review classifier, not a replacement for lexical composition. If that
hypothesis fails, token-level gaze is unlikely to be the first fix —
the signal may simply not be about sentiment.

---

## Why five features, not eight

The ZuCo joined table has eight measures. The MLP takes five:

```text
nFixations, FFD, GPT, TRT, GD
```

Dropped:

- **SFD** — mostly zero on anything re-read; a bad transfer target.
- **meanPupilSize** — lighting / baseline / subject anatomy. Dangerous
  to treat as “arousal = emotion” on 12 subjects.
- **omissionRate** — interesting, but highly correlated with length and
  with nFixations after the per-fixated-word normalisation.

Keeping GPT *and* TRT *and* GD looks redundant. It is. The 16-d linear
map can collapse them. I would rather let the layer do that than pick a
single “best” duration by hand on 400 rows.

---

## Why z-score on ZuCo and “whatever the CSV has” on full SST

400-row z-scoring puts every measured column on the same footing before
Adam sees it. Min-max would also work; the two files are the same join.

Full-SST gaze is already a model output. Re-z-scoring train/valid/test
separately would leak or shift. Re-z-scoring on train only is the
correct extra step if you retrain; the historical script does not.

Never train on min-max ZuCo and evaluate on z-scored ZuCo. They are
not a train/test pair.

---

## Why the toy example exists

The Hub trainers answer “does a 110M encoder plus 96 gaze weights beat
the same encoder?” That is the real question, and it needs a GPU.

A second question is cheaper and easy to lie about: “do these five
columns contain *any* label information?” If a logistic regression on
gaze-only is at 50%+ on 400 rows, you may be classifying review
*length* or *difficulty*, not emotion. The toy script reports:

- majority baseline
- length-only
- gaze-only
- hashed text-only
- hashed text + gaze

If text+gaze ≈ text-only and gaze-only ≈ majority, late fusion in
RoBERTa is unlikely to be a miracle. If gaze-only is strong, distrust
the feature (leakage, length, or a preprocessing bug) before celebrating.

---

## Why 20 epochs on 400 rows is a smell

Full fine-tune, batch 16, 20 epochs, no early stopping, five times.
The encoder can memorise 320 training sentences. Fold scores then
measure “did this seed memorise in a way that transfers to 80 rows?”
more than “did gaze help”.

Mitigations I would actually use:

- Freeze the encoder for 1–2 epochs, train the 16-d + classifier, then
  unfreeze with a smaller LR.
- Or use a 2-layer MLP on frozen `[CLS]` + gaze (linear probe + fusion).
- Or early-stop on a 40-row inner valid — noisy, but cheaper than 20
  full epochs.

The historical script does none of this. When you compare to it, compare
to *it*, not to a probe.

---

## Why predicted gaze on full SST is a different claim

Experiment A: humans read these 400 sentences.  
Experiment B: a model *guesses* how someone might have read 11k
sentences, then another model uses that guess as a feature.

A gain on B can mean:

1. The predictor recovered something like length / frequency / surprisal,
   and the classifier liked that extra channel.
2. The predictor leaked SST label information (trained on ZuCo reviews
   that overlap SST, or trained with a sentiment-aware encoder).
3. Noise that happened to regularise a 5-epoch run.

(1) is interesting as feature engineering. (2) is contamination. (3) is
a shrug. The write-up should not say “eye-tracking improves SST” for
Experiment B.

`examples/inspect_datasets.py` labels family B as `projected` for that
reason.

---

## Subject averaging vs keeping subjects

12 subjects are collapsed to one mean before any classifier. That:

- reduces missingness
- hides systematic readers (slow, skipping, regressing)
- makes it impossible to test “does fusion help for every reader?”

A next personal experiment: train 12 leave-one-subject-out logistic
probes on sentence-level raw `*_SR.csv` joined to `ssts_ZuCo.csv`.
`examples/subject_variability.py` only reports feature spread, it does
not train 12 models.

---

## What I am not claiming

- That gaze is emotion.
- That RoBERTa+gaze is SOTA on SST.
- That the Hub scripts are a library.
- That predicted `nFix` on full SST is a fixation count.

The claim the repo *can* support, after a careful re-run: **on this
400-sentence overlap, a 16-d late-fusion head did or did not change
stratified 5-fold metrics relative to the same encoder**. Everything
else is scaffolding.
