# Personal ablation plan

What to run, in order, the next time this repo sees a GPU. The
example suite already answers several questions **without** training.
Do not skip them: a fusion gain that example 11 cannot even hint at
is probably noise or a bug (see the last-batch test loop).

## Already answered locally

| Question | Where | Short answer |
| --- | --- | --- |
| Are the splits leak-free? | `07` | Yes, full SST 80/10/10 is disjoint and covers the parent table. |
| Is predicted gaze a standalone classifier? | `11` | No. Logreg loses to majority on valid/test. |
| Is measured ZuCo gaze a standalone classifier? | `11` | Barely. +3.5 accuracy over majority with the five fusion features. |
| Are ZuCo fusion features class-related after permutation? | `13` | No. GD is the least-null at perm p=0.12. |
| Is it just length? | `12` | Not in a simple way. Classes have similar length; ZuCo ET is negatively correlated with tokens because of per-word averaging. |
| Can I trust `combined_sst_et_*.csv` after sentence 149? | `10` | Only after remapping subject 3. The checked-in average did not. |

## GPU ablations (when you train)

Run these as **pairs** on the same seed / same folds. Change one thing.

### A. Text-only vs fusion on ZuCo 400 (the clean question)

Protocol: `model_ZuCo_SST.py` StratifiedKFold(5), 20 epochs, batch 16.

1. `roberta`
2. `roberta_eye_tracking`
3. `bert`
4. `bert_eye_tracking`

Report fold mean ± std of accuracy and weighted F1. A 1–2 point gaze
gain is compatible with example 11. A 10-point gain is a red flag
(overfit to 320 training rows, or a script bug).

Optional: freeze the encoder for two epochs, then unfreeze — not
implemented; would need a code change.

### B. Same four `model_type`s on full SST

Protocol: `model_full_SST.py` holdout, 5 epochs, batch 256. **Patch the
test loop** (`extend` preds) before quoting test numbers. Checkpoint on
accuracy is fine; also log weighted F1.

If fusion wins here, remember the gaze columns are predicted. Combine
with example 11: predicted gaze is not a linear classifier, so a
fusion win would be a *nonlinear* use of a weak extra 16-d vector.

### C. Shuffle-gaze negative control (not implemented in the training
scripts)

On ZuCo, permute the five gaze columns **across rows** (keep text and
label aligned, break gaze–label). Fusion should fall back to the
text-only number. If shuffled gaze still "helps," the head is using
garbage capacity.

### D. Sixth-channel experiment

Example 03 liked **SFD** on ZuCo (F=4.03). The training scripts do not
use it. A personal experiment: `num_eye_tracking_features = 6` and
add `SFD`. Example 06 `--num-gaze-features 6` checks concat width first.

Pupil size is a different construct (arousal). Try it only as a
separate ablation, not mixed into the five reading-time features
without a note.

### E. Do not bother until…

- `models/` exists (the full-SST script will not mkdir).
- You have written down git commit + `model_type` + device.
- You have re-run `07` and `10` on the CSVs you will actually load.

## Reporting template (personal log)

```
date:
commit:
script: model_ZuCo_SST.py | model_full_SST.py
model_type:
device:
epochs / batch / lr:
gaze: measured ZuCo | predicted full SST | shuffled
val or fold mean acc / f1:
test (only if test loop patched):
notes:
```

Keep that in a local file or in `docs/` if you want it in git. This
repo does not currently store training logs.
