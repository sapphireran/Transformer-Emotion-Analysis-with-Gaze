# Findings from the personal example suite

Numbers below were produced by `examples/run_all.py` plus examples
11–14 on this checkout. They are **not** transformer test scores.
Re-run the suite if you change a CSV.

Related scripts: `02` majority, `03` ANOVA, `10` subject-3 alignment,
`11` gaze-only logreg, `12` length residual, `13` label shuffle, `14`
sentence walkthrough.

## 1. Class balance and the majority floor

| Track | n | neg / neu / pos | Majority class | Majority accuracy | Majority weighted F1 |
| --- | ---: | --- | --- | ---: | ---: |
| Full SST combined | 11,853 | 4,649 / 2,241 / 4,963 | positive | 0.419 | 0.247 |
| Full SST train | 9,482 | 3,710 / 1,833 / 3,939 | positive | ~0.415 | — |
| Full SST valid | 1,185 | 476 / 209 / 500 | positive | 0.422 | — |
| Full SST test | 1,186 | 463 / 199 / 524 | positive | 0.442 | — |
| ZuCo 400 | 400 | 123 / 137 / 140 | positive | 0.350 | 0.181 |

The unused ZuCo 320/40/40 files are disjoint, but
`model_ZuCo_SST.py` ignores them. Whitespace token length is ~19 on
full SST and ~18 on ZuCo; polarity classes are within one token of
each other, so length is a weak label proxy.

## 2. Gaze-only logistic regression (no BERT)

Scaled multinomial logreg, same five features the fusion head sees.

| Setup | Accuracy | Weighted F1 | vs majority |
| --- | ---: | ---: | ---: |
| ZuCo fusion5, 5-fold CV | 0.385 | 0.373 | **+0.035** |
| ZuCo all ET columns | 0.395 | 0.380 | +0.045 |
| ZuCo length only | 0.315 | 0.266 | −0.035 |
| Full SST fusion5 → valid | 0.414 | 0.359 | **−0.008** |
| Full SST length → valid | 0.414 | 0.279 | −0.008 |
| Full SST fusion5 → test | 0.437 | 0.387 | −0.005 |
| Full SST length → test | 0.449 | 0.326 | +0.008 |

Personal reading:

- On **measured** ZuCo gaze, five numbers beat majority by about three
  points. That is a real but small side channel. Adding
  omissionRate / pupil / SFD adds one more point.
- On **predicted** full-SST gaze, the same model **loses** to majority
  on both holdout splits. Length-only is essentially tied. Do not
  tell a story in which the predicted columns are "how humans read
  positive reviews."
- A transformer that concatenates those five numbers can still
  ignore them. Example 11 does not prove fusion is useless; it proves
  the gaze vector is not a standalone sentiment classifier.

## 3. ANOVA vs permutation

sklearn `f_classif` on n=11,853 will call tiny mean shifts
"significant." Example 13 shuffles labels 200 times.

**Full SST predicted gaze**

| Feature | Observed F | Permutation p (200) |
| --- | ---: | ---: |
| GD | 32.10 | 0.00 |
| FFD | 27.71 | 0.00 |
| nFix | 27.56 | 0.00 |
| GPT | 26.78 | 0.00 |
| TRT | 24.85 | 0.00 |

The class-conditional means really do differ. Combined with example
11, that difference is **not large enough** to beat a majority
classifier. Large-n F is not an accuracy.

**ZuCo 400 measured gaze (fusion five)**

| Feature | Observed F | sklearn p | Permutation p (200) |
| --- | ---: | ---: | ---: |
| GD | 2.21 | 0.111 | 0.12 |
| FFD | 1.08 | 0.339 | 0.33 |
| nFixations | 0.96 | 0.384 | 0.45 |
| TRT | 0.83 | 0.438 | 0.49 |
| GPT | 0.39 | 0.678 | 0.71 |

None of the five fusion features survive a permutation check on 400
sentences. Example 03's extra columns (SFD F=4.03, p=0.019) are the
only ZuCo ET numbers that look class-related, and they are **not** in
`model_ZuCo_SST.py`.

## 4. Length confound

Per-class token means are close (full SST: 19.2 / 18.6 / 19.5;
ZuCo: 17.3 / 18.0 / 18.1). Pearson r of ZuCo fusion features with
token count is **negative** (TRT −0.55, GPT −0.54, nFixations
−0.47): longer sentences have *lower* per-word averages, which is
exactly how `DataTransformer` normalizes (divide by nwords_fixated).

Residualizing on length does **not** create a ZuCo effect (GD F
goes 2.21 → 2.64, still permutation-weak). On full SST, GD is almost
uncorrelated with length (r=0.018) and keeps F≈32 after residualizing
— it is the least length-like of the predicted five.

## 5. Subject 3 reindex

`3_SR.csv` has 299 rows. `SentLen` vs subject 1:

- ids 0–149: match rate 1.0
- ids 150–298 as stored: match rate 0.013
- after remap 150..298 → original 250..398: match rate 1.0

Row-wise averages in `average_data.csv` mix sentences from id 150
on. Example 04 remaps before CV; the training CSV does not.

## 6. Word-level quirks the fusion head never sees

Walked ids: 0 (neutral hedge), 3 (short "Slow, silly…"), 4
(negative communion-wafer line), 316 (stub "…pitiful, slapdash
disaster.").

- Sentence 0, "decency" has GPT ≈ 1304 ms — a late wrap-up / wrap-back
  that sentence-level GPT (z=1.59) is summarizing.
- Sentence 3, "unintentionally" (15 chars) takes ~5.4 fixations and
  802 ms TRT; "hilarious" has GPT ≈ 1033 ms. Short sentence, huge
  late measure — this is why sentence averages need the per-word
  view.
- Sentence 4 word 2 is stored as **`emp11111ty`** instead of `empty`.
  That is a ZuCo token-cleaning bug, not English. `a` is skipped.
- Sentence 316 word 0 is `unknown` with all zeros: the ellipsis was
  stripped by the letter-only tokenizer. Skip rate on this 4-token
  stub is why it topped example 05's skip list.

The fusion head only receives the five sentence numbers. None of
these word stories enter `EyeTrackingModel`.

## 7. Predicted vs PROVO shape

PROVO (2,659 words, 134 sentences) still has `fixProp`. Predicted SST
v2 covers all 11,853 sentences / 191,971 letter-tokens. Predicted
`nFix`–`TRT` r=0.96 (PROVO 0.98) but predicted `nFix`–`FFD` is only
0.36 (PROVO 0.90). The predictor copies the nFix–TRT ridge and
smears first-fixation duration. `result/*.png` is the picture of
that; example 08 is the table.

## What this implies for a later training run

1. Quote **validation** numbers from `model_full_SST.py`, not the
   printed test line (last-batch bug).
2. On ZuCo, a `roberta` vs `roberta_eye_tracking` ablation is a test
   of a **weak** side channel. Do not expect a large gain.
3. If full-SST fusion beats text-only, check that it is not just the
   model using predicted GD as a tiny extra bit that logreg could not
   use linearly — and remember those columns are not human gaze.
4. If you add a sixth channel, SFD is the ZuCo feature example 03
   actually liked. It is also sparse. See [ablation_plan.md](ablation_plan.md).
