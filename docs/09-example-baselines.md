# Example baselines (from this checkout)

Numbers below come from `python3 examples/run_all.py` on the committed
CSVs. They are **personal lab notes**, not paper results. Re-run the
scripts if you change a table.

## Linear late fusion on the 400 (`examples/06_toy_late_fusion.py`)

Protocol: stratified 5-fold, seed 42, softmax + L2, no transformer.

| Method | Accuracy | Weighted F1 |
| --- | --- | --- |
| majority (train-fold mode) | 0.350 ± 0.004 | 0.182 ± 0.004 |
| gaze (5-d, z-scored in-fold) | 0.390 ± 0.065 | 0.367 ± 0.064 |
| text-stats (7 hand features) | 0.448 ± 0.047 | 0.429 ± 0.059 |
| concat text-stats + gaze | 0.470 ± 0.037 | 0.458 ± 0.048 |

Takeaways that are safe:

- Gaze-only is **above majority** but noisy fold-to-fold (one fold
  hits 0.51 acc, two sit at ~0.33–0.35). The 5-d vector is not
  random, and it is not a stable classifier by itself.
- The tiny polarity lexicon + length features beat gaze. That is
  expected: SST labels *are* about words.
- Concatenation is a bit better than text-stats alone (~2 points).
  That is the most you can claim from this toy: the two views are
  not identical. Confirm with `model_ZuCo_SST.py` before talking
  about RoBERTa.

Pooled-fold confusion for concat still dumps a lot of negatives into
neutral/positive (42 / 46 / 35). Gaze-only barely finds negatives
(20 / 50 / 53). If fusion ever helps the transformer, watch the
**negative** class, not the headline accuracy.

## Class-conditional gaze on the 400 (`examples/03_gaze_feature_summary.py`)

Standard-scaled means sit at 0 by construction. Class means are
small (tenths of a standard deviation):

| Class | nFixations | FFD | GPT | TRT | GD |
| --- | ---: | ---: | ---: | ---: | ---: |
| negative | −0.09 | −0.08 | −0.06 | −0.10 | +0.03 |
| neutral | +0.08 | −0.03 | +0.04 | +0.05 | −0.14 |
| positive | −0.00 | +0.10 | +0.01 | +0.03 | +0.11 |

Fusion columns are **highly collinear**: nFix–TRT *r* = 0.96,
nFix–GPT = 0.91, TRT–GPT = 0.94. FFD is the most independent of
the five (*r* ≈ 0.42 with nFix). A 16-d linear projection of five
near-duplicates is mostly a 2-d or 3-d signal.

## Length is not the label (`examples/04_label_and_length.py`)

On the 400, mean whitespace tokens are 17.6 / 18.3 / 17.6 for
neg / neu / pos. Full-SST train is similar (19.2 / 19.2 / 19.1).
A “longer reviews are more negative” story does not hold here.
Correlations of the five gaze columns with token count are also
weak on the **already aggregated** sentence table (the interesting
length effect lives at the **word** level).

## Word-level regularities (`examples/05_word_level_gaze.py`)

7,129 tokens, 400 sentences, 1.3% never-fixated (`nFixations == 0`).

| WordLen | n | mean nFix | mean TRT (ms) |
| --- | ---: | ---: | ---: |
| 1–2 chars | 1359 | 0.47 | 53 |
| 3–4 | 2459 | 0.83 | 95 |
| 5–6 | 1503 | 1.31 | 156 |
| 7–9 | 1234 | 1.71 | 205 |
| 10+ | 519 | 2.21 | 270 |

`corr(WordLen, nFixations) = 0.75`, `corr(WordLen, TRT) = 0.73`.
A closed-class stub list (the, of, to, …) has mean TRT 64 ms vs
166 ms for everything else. Sentence-level fusion **washes this
out** by averaging over fixated words.

`0_NR` matches SST sentence 0 (neutral): *“Presents a good case
while failing …”* — `failing` and `presents` get ~3 fixations and
~380 ms TRT; `us` is almost skipped. That is the structure a
word-aligned model could use and the current `pooler_output` +
5-d vector cannot.

## Readers do not agree that much (`examples/07_subject_variance.py`)

On the 299 sentences every subject has:

- Mean pairwise TRT correlation ≈ **0.41** (min 0.07, max 0.59).
- Subject 3 is the odd one out (many *r* < 0.25).
- Per-sentence CV across readers: FFD 0.15, GD 0.21, nFix 0.25,
  TRT 0.29, GPT 0.33.

Averaging is therefore a **denoise and a blur**. GPT (the
regression-path measure) is the noisiest across people and also
almost a duplicate of TRT after averaging — a candidate to drop
in an ablation.

`average_data.csv` matches a plain 12-reader mean on aligned TRT
to ~0.25 ms MAE. The 0→NaN step in the averaging script barely
moves the mean on this checkout.

## Splits are clean partitions (`examples/08_split_integrity.py`)

- ZuCo leftover 320/40/40: no overlap, covers all 400 ids.
- Full SST 9482/1185/1186: no overlap, covers all 11,853 ids.
- `ssts_ZuCo.csv` and the two combined gaze tables share the same
  id set; standard and min-max are in the same order.

Still true: `model_ZuCo_SST.py` ignores the leftover 320/40/40 and
re-folds. The integrity script only certifies the files.

## Schema contracts

`examples/02_schema_check.py` passed on this checkout: headers, row
counts, `{0,1,2}` labels, and parseable fusion floats. That is the
regression test to keep green when you regenerate CSVs.
