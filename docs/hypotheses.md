# Hypotheses

Personal notes on what this architecture could learn, written after
looking at the tables rather than after a GPU run.

## H1 — Gaze is not a standalone sentiment classifier

On the 400-row z-scored table, Pearson r between each fusion feature
and the integer label is below 0.08. A gaze-only linear readout
beats majority (≈0.46 vs 0.35 on an 80-row stratified holdout) but
does not look like a sentiment model. If `roberta_eye_tracking`
beats `roberta`, the interesting part is the *interaction* with the
pooler, not a hidden polarity axis in TRT.

## H2 — The five fusion channels are closer to two

nFixations, GPT, and TRT share r > 0.91. GD is a slightly different
direction; FFD is the most independent of the five and also the
measure with the weakest inter-reader r (0.10). A 5→16 layer can
re-weight this, but an ablation that drops GPT or nFixations is
likely to move the score less than an ablation that drops FFD.

## H3 — Averaging twelve readers hides the disagreement

Mean pairwise r on TRT is 0.44 even on the clean 0–149 window.
Pupil size has pairs below 0. Some of the "gaze signal" in the
sentence table is a compromise between people who lingered and
people who skipped. A subject-aware model (or leaving reader 3's
remapped rows *out* of the mean) is a different experiment than
the one this clone runs.

## H4 — Full SST tests a different claim

Projected gaze on 11,853 sentences can improve a classifier by
leaking **text-derived** structure back in, even if no human ever
looked at those reviews. A fair "does human gaze help?" comparison
belongs on the 400-row track. A fair "does a gaze-shaped extra
channel help at scale?" belongs on full SST — and only after the
test loop is aggregating every batch.

## H5 — Neutral is the fragile class at scale

Full SST is 4649 / 2241 / 4963. Weighted F1 can look healthy while
neutral recall is poor. The ZuCo overlap is almost balanced, so a
number that transfers from 400 to 11k without a per-class table is
not telling you the same story.

## H6 — Sentence-level fusion cannot see the word that hurt

The word table for sentence 0 shows `failing` (nFixations 3.08,
TRT 387) and `us` (nFixations 0.08, TRT 12) in the same review.
The model sees one z-scored 5-tuple. If the scientific question is
"do people dwell on sentiment-bearing words?", this head is the
wrong instrument. That question wants word-aligned features or a
token-level gate, which this clone does not implement.

## What I would run next (personal, not done here)

1. Fix the test loop, then re-report full-SST test metrics.
2. Re-average the twelve readers with subject 3 remapped; train the
   5-fold script on both averages and diff the mean F1.
3. Ablate `{nFixations, TRT, GPT}` down to `{FFD, GD}` on ZuCo.
4. Keep the 400-row track as the only place a paper sentence says
   "human eye-tracking."
