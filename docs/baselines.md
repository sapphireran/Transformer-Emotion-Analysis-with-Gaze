# Recorded dummy baselines

Numbers from `python3 examples/dummy_baseline.py` on this checkout (Python 3.12, scikit-learn 1.9). They are floors for the transformer runs, not model results.

Re-run the script before quoting them in a new environment. The ZuCo folds use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` — the same splitter as `model_ZuCo_SST.py`. SST uses the checked-in `train_full_sst.csv` / `test_full_sst.csv`.

## ZuCo (400 sentences, measured gaze)

Mean over 5 folds:

| model | accuracy | weighted P | weighted R | weighted F1 | macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority (always positive / 2) | 0.3500 | 0.1225 | 0.3500 | 0.1815 | 0.1728 |
| gaze-only logistic regression | 0.3675 | 0.3597 | 0.3675 | 0.3535 | 0.3489 |

Per-fold gaze-only accuracy: 0.3875, 0.4125, 0.3625, 0.3250, 0.3500.

Gaze-only is a couple of points above majority on average and loses fold 4. That is the “weak side channel” pattern: not zero, not something you would ship without text.

## SST holdout (transferred gaze)

| model | accuracy | weighted P | weighted R | weighted F1 | macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority (always positive / 2) | 0.4418 | 0.1952 | 0.4418 | 0.2708 | 0.2043 |
| gaze-only logistic regression | 0.4368 | 0.3576 | 0.4368 | 0.3870 | 0.3065 |

Gaze-only **loses to majority on accuracy** and never predicts neutral on the test set (confusion matrix: 0 column for class 1). It does pick up some negative/positive structure, which is why weighted and macro F1 rise.

SST test confusion (gaze-only):

| gold \ pred | negative | neutral | positive |
| --- | ---: | ---: | ---: |
| negative | 163 | 0 | 300 |
| neutral | 76 | 0 | 123 |
| positive | 169 | 0 | 355 |

## How to use these

- A ZuCo `roberta_eye_tracking` mean CV accuracy that cannot beat **0.37** is not using the encoder.
- An SST fusion accuracy near **0.44** with a dead neutral class is majority-shaped; check the last-batch test bug in [known-issues.md](known-issues.md) before writing it down.
- Gaze-only F1 on SST (0.387 weighted) is the number a text model must beat *as a classifier*, not the number fusion must beat by a huge margin. Fusion is supposed to add a little on top of RoBERTa, not on top of this logreg.

See also [models-and-training.md](models-and-training.md) and `examples/dummy_baseline.py`.
