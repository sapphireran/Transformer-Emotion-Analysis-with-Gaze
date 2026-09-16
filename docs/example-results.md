# Example-script results on this checkout

Numbers below were produced by `examples/scripts/*.py` against the committed CSVs. They are a baseline for later personal runs, not a paper table.

## 400-row ZuCo SST (`01`, `02`)

| Item | Value |
| --- | --- |
| Rows | 400, ids and labels match `ssts_ZuCo.csv` |
| Class prior | NEG 123 (30.75%) / NEU 137 (34.25%) / POS 140 (35.00%) |
| Majority floor | **0.3500** |
| Whitespace tokens | min 3, mean 17.82, max 43 |
| Missing values on ET columns | none |
| TRT vs nFixations *r* | **0.9581** |
| GPT vs nFixations *r* | 0.9125 |
| FFD vs nFixations *r* | 0.4213 |

Class-conditional means of the z-scored five channels are all **< 0.25 σ** apart. The largest gap is GD, neutral vs positive (`|Δmean|/std ≈ 0.25`). Linear gaze is a weak sentence-level sentiment feature on this set. That matches the hypothesis in `docs/overview.md`: any transformer-fusion win has to come from a tiny residual, not from a clean gaze–label plane.

## Logistic CV (`03`, 5-fold, seed 42, scaler inside the fold)

| Model | Acc | Weighted F1 | Macro F1 |
| --- | --- | --- | --- |
| Majority | 0.3500 ± 0.0000 | 0.1815 ± 0.0000 | 0.1728 ± 0.0000 |
| Length only (`n_tokens`) | 0.3350 ± 0.0271 | 0.2568 ± 0.0162 | 0.2475 ± 0.0127 |
| Gaze only (5 channels) | **0.3675 ± 0.0338** | 0.3535 ± 0.0314 | 0.3489 ± 0.0290 |
| Gaze + length | 0.3475 ± 0.0409 | 0.3411 ± 0.0397 | 0.3376 ± 0.0398 |
| Gaze residualized on length | 0.3500 ± 0.0342 | 0.3441 ± 0.0320 | 0.3411 ± 0.0322 |

Reading: gaze-only is about **1.8 points** above majority and inside fold noise (`±3.4`). Residualizing on length puts accuracy back on the majority line. I would not spend a GPU night on fusion expecting a large gaze lift until a word-aligned head exists.

## Convenience splits (`04`)

ZuCo `train.csv` / `valid.csv` / `test.csv` (320 / 40 / 40):

- Id-disjoint, cover all 400 `sentence_id`s.
- **Not stratified.** Valid is 47.5% positive vs 35% in the combined table. Max class drift **0.1325**.
- `model_ZuCo_SST.py` does not use these files (it re-folds). Example baselines should prefer CV or the stratified helper.

Full SST `*_full_sst.csv` (9,482 / 1,185 / 1,186):

- Id-disjoint.
- Class drift vs train ≤ **0.0264**.
- Neutral is the minority (~17–19%). Weighted metrics will look kinder than macro-F1.

## Word-level averages (`05`)

| Bin | nFixations | GD | TRT | n words |
| --- | --- | --- | --- | --- |
| 1–2 | 0.45 | 42 | 52 | 1,414 |
| 3–4 | 0.83 | 73 | 95 | 2,459 |
| 5–6 | 1.31 | 110 | 156 | 1,503 |
| 7–8 | 1.63 | 135 | 196 | 925 |
| 9–12 | 2.04 | 164 | 247 | 743 |
| 13+ | 2.69 | 218 | 329 | 85 |

Monotone and unsurprising: longer words get more fixations and more time. `unknown` tokens are 0.77% of 7,129 rows.

The highest-TRT list is dominated by **glued tokens** from punctuation stripping (`murderoncampus`, `allwiseguysallthetime`, `unintentionally` is real; `emp11111ty` is a dump artifact). Sentence means inherit those spikes. See [limitations.md](limitations.md).

Scanpath for `sentence_id=0` (neutral): function words (`a`, `to`, `us`, `the`) sit well below 1 fixation; `failing`, `reason`, `presents` sit at 2.5–3. `decency` has an enormous GPT (1304 ms) — wrap-up / end-of-sentence.

## Fusion shapes (`06`)

Locked: pooled `(B, 768)`, ET `(B, 5)`, gaze hidden `(B, 16)`, fused `(B, 784)`, logits `(B, 3)`. A dummy forward on the first 8 real ET rows produces finite logits.

## How to refresh this page

```bash
python3 examples/scripts/01_explore_zuco_sst.py
python3 examples/scripts/02_gaze_feature_report.py
python3 examples/scripts/03_gaze_only_baseline.py
python3 examples/scripts/04_split_sanity_check.py
python3 examples/scripts/05_word_level_scanpath.py
python3 examples/scripts/06_fusion_shape_check.py
```

If a later commit regenerates the combined CSVs, replace the tables above in the same commit.
