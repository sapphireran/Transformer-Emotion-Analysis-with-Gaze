# Predicted-gaze degeneracy

`EyeTrackingModel` allocates a `Linear(5, 16)` stem as if the five gaze
channels were five numbers. On recorded ZuCo they are correlated; on
projected full SST they are almost one number.

## Correlation condition number

Computed on the committed combined tables, five fusion columns, Pearson
correlation matrix, ratio of largest to smallest eigenvalue:

| table | n | cond(C) | smallest eigenvalue |
| --- | ---: | ---: | ---: |
| ZuCo ∩ SST, recorded | 400 | ≈ 304 | ≈ 0.013 |
| full SST, projected | 11,853 | ≈ 13,733 | ≈ 3.3e-4 |

On full SST, `nFix` / `FFD` / `GPT` / `TRT` have pairwise r ≥ 0.986. `GD`
is the only channel that is merely strongly related (r ≈ 0.68–0.78). A
16-d projection of that vector is a 16-d projection of one latent plus
noise.

## Label association is tiny

Pearson r against `{0,1,2}` on ZuCo sits between 0.03 and 0.07. On full
SST the five projected channels are all around −0.05. That is not “gaze
encodes polarity.” It is “gaze is almost orthogonal to the label, and on
SST it is also almost rank-1.”

## Length confounding

ZuCo sentence nFixations is a *per-fixated-word* mean. Longer reviews
have a lower mean nFixations (r ≈ −0.47 with whitespace word count). Full
SST projected `nFix` still correlates with length (r ≈ −0.30). A gaze-only
linear model can pick up “this review is short” rather than “the reader
struggled with the valence.” `examples/08_cpu_baselines.py` residualizes
the five channels against word count before a ridge probe.

## Mean-pooling `prediction_test_v2.csv`

191,971 word rows, 11,853 sentence ids, values on a roughly 0–100
predictor scale (nFix roughly 15–43). The sentence table is z-scored
(mean ≈ 0). Mean-pooling the word predictor versus `combined_full_sst_et`
`nFix` gives Pearson r ≈ 0.58 — related, not a reconstruction, and the
wrong scale for `EyeTrackingModel` as wired. Do not drop the word file
into the sentence trainer.

## Practical consequence

If a full-SST fusion run beats text-only by a few tenths of a point, the
first suspicion is length / collinear leakage, not a rich five-channel
cognitive signal. The ZuCo track is the only place in this clone where
the five names are five measured quantities.
