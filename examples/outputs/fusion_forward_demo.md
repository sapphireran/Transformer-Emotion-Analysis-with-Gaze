# Late-fusion CPU demo (ZuCo)

| model                                 | train acc | holdout acc | holdout macro F1 | final loss |
|---------------------------------------|-----------|-------------|------------------|------------|
| majority                              | —         | 0.3500      | 0.1728           | —          |
| gaze-only                             | 0.4033    | 0.4300      | 0.4287           | 1.2179     |
| text-only (hash)                      | 0.5000    | 0.3300      | 0.3264           | 1.0391     |
| concat-linear (hash ⊕ gaze)           | 0.5300    | 0.4200      | 0.4156           | 1.0807     |
| fused two-layer (hash ⊕ Linear(gaze)) | 0.4967    | 0.3900      | 0.3660           | 0.9960     |
| fused two-layer + shuffled gaze       | 0.5033    | 0.3600      | 0.3360           | 1.0076     |

Confusion matrix for the two-layer fused head on the holdout:

|               | pred negative | pred neutral | pred positive |
|---------------|---------------|--------------|---------------|
| true negative | 6             | 8            | 17            |
| true neutral  | 1             | 11           | 22            |
| true positive | 7             | 6            | 22            |

- 400 ZuCo rows, stratified holdout 100 (25%), seed=42, epochs=35.
- Text vector is a 32-d hashed bag of words, not RoBERTa.
- Gaze input is the z-scored 5-d set (nFixations, FFD, GPT, TRT, GD).
- concat-linear is one softmax on [text; gaze]. The two-layer fused head matches EyeTrackingModel: Linear(gaze) then concat then Linear.
- If shuffled-gaze ≈ fused, the gaze branch is not carrying signal in this tiny head. If it is worse, the real gaze values were used.
- Do not paste these accuracies next to GPU transformer runs.
