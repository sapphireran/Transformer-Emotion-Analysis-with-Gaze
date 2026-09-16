# Findings from the committed tables

Numbers below are what `examples/run_all.py` is designed to regenerate.
If a later CSV rewrite changes them, trust the outputs folder.

## Label priors

Full SST combined (11,853): negative 4649, neutral 2241, positive 4963.
Train majority baseline (always positive) ≈ **0.415** accuracy.

ZuCo-SST (400): negative 123, neutral 137, positive 140. Majority ≈ **0.35**.
The leftover `valid.csv` (40 rows) has only 7 negatives — do not model-select
on it.

## Gaze is weakly tied to polarity

Full SST train, Pearson r(feature, `sentiment_label`):

| nFix | GD | TRT | FFD | GPT |
| --- | --- | --- | --- | --- |
| ≈ −0.066 | −0.070 | −0.062 | −0.065 | −0.065 |

ZuCo-SST combined, same idea on the five model columns: |r| mostly 0.03–0.07
(FFD is the largest, still < 0.08).

Class-conditional means on **z-scored** full SST: neutrals have the
*highest* mean nFix/GD/TRT; positives sit below zero. Residualizing against
whitespace token count barely moves r(feature, label). Length is not the
hidden explanation of the weak polarity link.

A row-shuffle of gaze (`examples/10_shuffle_control.py`) drives those
correlations to ~0, as it should. The pattern is real and small.

## The sidecar is almost one-dimensional

Full SST train SVD on the five z-scored columns: PC1 ≈ **91.7%**, PC2 ≈
**8.2%** (GD), remainder dust. nFix/TRT/FFD/GPT r > 0.98. See
[sidecar-geometry.md](sidecar-geometry.md).

## Predicted gaze anti-correlates with length

On full SST train, r(ntokens, nFix) ≈ **−0.30**, r(ntokens, TRT) ≈ **−0.37**,
r(ntokens, GD) ≈ 0.01. Measured ZuCo sentence features do not show that
strong anti-correlation. Treat Track B gaze as a **model output**, not as
milliseconds.

## Subject 3

- 299 sentences, ids 0–298
- packed 150 = original 250 (`SentLen` and tokens both agree)
- word-level skip rate ≈ **0.525** vs ~0.21–0.39 for the other readers
- mean omission rate is the highest of the twelve
- positional averages after the hole are mixed reviews

## Splits

Full SST ids partition the combined table. Two review *strings* leak
(train/valid and train/test). ZuCo vs full SST share **three** strings.
`model_ZuCo_SST.py` ignores `{train,valid,test}.csv`.

## Test-loop coverage

1186 / 256 → last batch 162 rows. Last-batch label prior ≠ full test prior.
Any published-looking `Test Acc` from an unpatched `model_full_SST.py` run
is that window.

## Pupil outlier

Subject 12 mean pupil ≈ 298 vs ~530–1350 for the others. It is in the
positional sentence averages with equal weight.

## Empty tokens

Subject 1 word table: 55 rows with `Word` null and `WordLen == 0`
(punctuation stripped to nothing). Harmless for sentence means; they
become `unknown` in `word_averages_v2.csv`.

## Overlap of fusion capacity vs signal

16 / 784 ≈ 2% of the concatenated vector is gaze, and that 2% is ~91% one
axis of a weakly labeled sidecar. The architecture can *look* multimodal
while the optimization mostly fits RoBERTa. The cheap CPU controls
(shuffle, PCA, last-batch, subject-3) are the things to rerun before
quoting a GPU ablation.
