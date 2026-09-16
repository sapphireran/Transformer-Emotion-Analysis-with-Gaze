# Worked example: from a ZuCo row to a fusion logit

Numbers below come from this checkout (the same CSVs the unit tests read).
No GPU and no `transformers` download.

## 1. Pick a sentence

`ZuCo_SST_data/combined_sst_et_standard.csv`, `sentence_id = 0`:

| Field | Value |
| --- | --- |
| sentence | Presents a good case while failing to provide a reason for us to care beyond the very basic dictums of human decency. |
| `sentiment_label` | `1` (neutral) |
| `nFixations` | `+1.5209` |
| `FFD` | `−0.0784` |
| `GPT` | (see CSV) |
| `TRT` | `+1.2480` |
| `GD` | `+0.0758` |

The five model channels are already z-scored. A linear gaze-only classifier
would see a slightly high fixation count and total reading time, which is
weakly associated with the neutral class in this table (neutral mean
`nFixations = +0.083`, negative `−0.089`). The gap is much smaller than the
within-class std (~1.0), so gaze alone is not enough.

`examples/join_check.py` confirms this row is exactly
`ssts_ZuCo.csv[0]` plus `standard_scaled_average_data.csv[id=0]`.

## 2. Confirm the split story

The 400 rows are **not** trained via `train.csv`. `model_ZuCo_SST.py` draws
`StratifiedKFold(..., random_state=42)`. The unused hold-out is 320 / 40 / 40
and the 40-row valid file has only 7 negatives — that is why CV is the
documented protocol.

`examples/split_integrity.py` still checks that those three files are a
disjoint cover, so you can switch to hold-out later without a silent leak.

## 3. Rebuild the gaze scaling

`examples/scaling_check.py` refits min-max and population z-score on
`average_data.csv`. On this checkout:

```text
min-max max|delta| = 3.6e-16
z-score max|delta| = 7.3e-15
```

So the checked-in scaled tables are ordinary sklearn-style column scaling,
not a custom transform.

## 4. Reconstruct sentence nFixations from words

For subject 1, `DataTransformer` averages word `nFixations` over words that
have any non-zero gaze field. `examples/word_to_sentence.py` repeats that
rule on `word/1_SR.csv` and matches `1_SR.csv` on all 400 sentences
(`max |delta| = 0`).

That is the contract you need if you ever pool `prediction_test_v2.csv` the
same way.

## 5. Fake a forward pass

`examples/fusion_forward.py` does not load RoBERTa. It draws a random pooler
`(B, 768)` and gaze `(B, 5)`, then applies:

```text
Linear(5 → 16) → concat → Linear(784 → 3)
```

A seed-0 batch of 8 yields logits shaped `(8, 3)`. Replacing gaze with zeros
moves the logits (`mean |Δ| ≈ 0.07` with the demo init). That is the whole
architectural claim: gaze is a 16-D side channel, not an attention prior.

## 6. Commands

```bash
python3 examples/join_check.py
python3 examples/scaling_check.py
python3 examples/word_to_sentence.py
python3 examples/fusion_forward.py --seed 0 --batch-size 8
python3 examples/gaze_by_sentiment.py
python3 -m unittest discover -s examples/tests -v
```

All of those returned 0 on the revision that added this page.
