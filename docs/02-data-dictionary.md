# Data dictionary

Column names drift across folders. The fusion head only ever sees five
numbers plus token ids.

## Sentiment

| value | meaning | ZuCo 400 | full SST |
| ---: | --- | ---: | ---: |
| 0 | negative | 123 | 4,649 |
| 1 | neutral | 137 | 2,241 |
| 2 | positive | 140 | 4,963 |

`convert_full_SST.py` maps folder names `NEGATIVE/NEUTRAL/POSITIVE` to
those integers. The headerless dump `SST_data/stts_all_sentence_level.csv`
still has the words `NEGATIVE` / `NEUTRAL` / `POSITIVE` in column 2 — read
it with `header=None` or the first review becomes a column name and the
count drops from 11,853 to 11,852.

## Gaze channels the model concatenates

| ZuCo name | full-SST name | reading-research name | short definition |
| --- | --- | --- | --- |
| `nFixations` | `nFix` | number of fixations | how many times the eyes landed (here: mean per fixated word) |
| `FFD` | `FFD` | first fixation duration | duration of the first landing |
| `GD` | `GD` | gaze duration | first-pass time on the word |
| `TRT` | `TRT` | total reading time | all fixations, including regressions |
| `GPT` | `GPT` | go-past time | time from first entry until the eyes move past the word |

On the 400-row z-scored ZuCo table, those five are correlated but not
interchangeable (`nFixations`–`TRT` r ≈ 0.96, `nFixations`–`FFD` r ≈ 0.42).
On the projected full-SST table they collapse: four channels have pairwise
r ≥ 0.986. See `docs/05-predicted-gaze-degeneracy.md`.

## Extra sentence fields (ZuCo only)

`SentLen`, `omissionRate`, `meanPupilSize`, `SFD` (single-fixation
duration) are exported by `DataTransformer` and survive into
`combined_sst_et_*.csv` except `SentLen`. None of them are passed into
`EyeTrackingModel`.

## Word-level tables

`ZuCo_et_csv_data/word/{1-12}_SR.csv` and `word_averages_v2.csv` add
`Sent_ID` (`{k}_NR` for normal reading), `Word_ID`, `Word`, `WordLen`.
`gaze_prediction/data/*.csv` use `sentence_id`, `word_id`, `word` and the
five-channel `nFix` naming.

## Id spaces

- ZuCo text / combined tables: `sentence_id` 0–399.
- Reader sentence files: `id` 0–399, except reader 3 which is 0–298
  **compacted**. Do not join reader 3 on `id` without
  `gazebook.remap.compact_to_original`.
- Word `Sent_ID` 150_NR on reader 3 is original sentence 250, not 150.
- Full SST `sentence_id` 0–11852 matches the headerless dump order.
