# Result plots

Three PNGs already live in `result/`. They are pair-plots of
word-level gaze, not model metrics. Example 08 reprints the same
story as numbers.

| File | What it shows |
| --- | --- |
| `result/provo_data_scatter_hist_plots.png` | PROVO extract (`gaze_prediction/data/provo.csv`, 2,659 words). Histograms on the diagonal, scatter of each pair. `nFix` is right-skewed around ~15; `FFD` is roughly 2.5–6; `TRT` vs `nFix` is almost linear. |
| `result/train_data_scatter_hist_plots.png` | Predicted gaze on whatever split was called "train" when the figure was drawn. Same five predicted columns as `prediction_test_v2.csv` (`nFix`, `FFD`, `GPT`, `TRT`, `GD`). |
| `result/test_data_scatter_hist_plots.png` | Predicted gaze on the matching "test" extract. |

## How to read them

- **PROVO** is measured human reading. Tight `TRT`–`nFix` and
  `GD`–`nFix` ridges are expected: more landings cost more time.
- **Predicted SST** copies that ridge for `nFix` vs `TRT` (r=0.96)
  but **not** for `nFix` vs `FFD` (r=0.36 vs 0.90 on PROVO). The
  predictor is better at total time than at first-fixation duration.
- Axes are not milliseconds in the predicted plots. `convert_zuco_data.py`
  min-max scaled some training features to 0–100. Compare **shape**,
  not the numeric range, to ZuCo `word/{1–12}_SR.csv` (raw ms).

## What they are not

- They are not confusion matrices, learning curves, or ablation bars.
- They do not include the five sentence-level columns that
  `EyeTrackingModel` actually concatenates.
- They do not prove that predicted gaze is interchangeable with ZuCo.

If you regenerate plots, keep them under `result/` and mention the
CSV + row count in the filename or in this note. Example 08 is the
reproducible numeric companion; the PNGs are snapshots.
