# Gaze-only baselines (ZuCo 400)

If the five fusion features already separated sentiment, a linear readout would show it. They do not, at least not on this z-scored table. That is useful: it means any gain from ``roberta_eye_tracking`` has to come from *interaction* with the text encoder, not from gaze as a standalone classifier.

| model | n | accuracy | weighted F1 |
| --- | --- | --- | --- |
| majority(2) | 400 | 0.3500 | 0.1815 |
| gaze_linear_ovr (stratified 20% holdout) | 80 | 0.4625 | 0.4391 |
| gaze_linear_ovr (in-sample, optimistic) | 400 | 0.4175 | 0.4051 |

## Protocol

Features: nFixations, FFD, GPT, TRT, GD. Holdout is stratified on the integer label with seed 42. The linear model is one-vs-rest least squares with a bias term, implemented in stdlib (no sklearn).
