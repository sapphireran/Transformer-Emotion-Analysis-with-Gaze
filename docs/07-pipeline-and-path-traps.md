# Pipeline and path traps

## Intended ZuCo sentence path

```
ZuCo .mat (not in git)
    → read_ZuCo_mat.py + DataTransformer(task1, sentence, scaling='raw')
    → et_csv_data/{1-12}_SR.csv          # script path
    → (committed as) ZuCo_et_csv_data/{1-12}_SR.csv
    → get_average_sentence_level.py      # 0 → NaN, index mean, then scale
    → average_data.csv
    → min_max_scaled_average_data.csv
    → standard_scaled_average_data.csv
    → join ssts_ZuCo.csv on sentence_id == id
    → combined_sst_et_{standard,min_max}.csv
```

`examples/06_join_and_scalers.py` checks the last join and both scalers.
Min-max rebuilds to ~1e-16. Standard scaling matches **population** std
(`ddof=0`, sklearn `StandardScaler`). Sample std (`ddof=1`) misses by
about 8e-3 and is the wrong fingerprint.

## Intended word path

```
DataTransformer(level='word')
    → ZuCo_et_csv_data/word/{1-12}_SR.csv
    → word/get_average.py                # again groupby(level=0)
    → word_averages_v2.csv
    → gaze_prediction/data/convert_zuco_data.py   # 0–100 min-max, nFix name
```

`convert_zuco_data.py` looks for `training_data/word_averages_v2.csv`,
which is not the committed path.

## Full SST path

```
stts_all_sentence_level.csv          # headerless text + POSITIVE/…
    → convert_sst_to_et.py           # NLTK tokens, zeros for gaze
    → sst_et_test.csv                # word-level placeholders
    → (external predictor, not in git)
    → sentence-level nFix/FFD/GPT/TRT/GD
    → combined_full_sst_et.csv
    → SST_data/spilt.py              # 80/10/10, seed 42, not stratified
```

Typo in the filename (`spilt.py`) is original.

## Path strings that will not run on this Linux checkout

| location | what it says | what exists |
| --- | --- | --- |
| `utils_ZuCo.get_matfiles` | `\\ZuCo_mat_data\\` | no MATLAB folder in git |
| `read_ZuCo_mat.py` | `et_csv_data/{i}_SR.csv` | `ZuCo_et_csv_data/` |
| `get_average_sentence_level.py` | `et_csv_data` | `ZuCo_et_csv_data/` |
| `gaze_prediction/data/convert_zuco_data.py` | `training_data/word_averages_v2.csv` | `ZuCo_et_csv_data/word/` |
| `convert_full_SST.py` | `ZuCo_SST_data/all/{NEG,…}` | those txt folders are not committed |

The committed CSVs are the source of truth for examples. Re-running the
export scripts needs the MATLAB drop and path fixes that are out of
scope for this personal notes layer.

## Duplicate helpers

`convert_full_SST.py` and `ZuCo_SST_data/save_SST_data.py` both walk
`NEGATIVE/POSITIVE/NEUTRAL` folders. The latter writes `output.csv` and
keeps `sentence_id` as a string. Prefer the committed `ssts_ZuCo.csv`.
