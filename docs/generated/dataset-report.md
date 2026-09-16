# Dataset report

Generated from the CSVs in this personal repository. Re-run
`python examples/08_write_dataset_report.py` after data changes.

## Checked-in tables

| key | file | kind | rows | gaze columns |
| --- | --- | --- | ---: | --- |
| `zuco_sst_standard` | `ZuCo_SST_data/combined_sst_et_standard.csv` | sentence | 400 | omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT |
| `zuco_sst_minmax` | `ZuCo_SST_data/combined_sst_et_min_max.csv` | sentence | 400 | omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT |
| `zuco_sst_text` | `ZuCo_SST_data/ssts_ZuCo.csv` | sentence | 400 | — |
| `zuco_sst_train` | `ZuCo_SST_data/train.csv` | split | 320 | omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT |
| `zuco_sst_valid` | `ZuCo_SST_data/valid.csv` | split | 40 | omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT |
| `zuco_sst_test` | `ZuCo_SST_data/test.csv` | split | 40 | omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT |
| `zuco_sentence_raw` | `ZuCo_et_csv_data/average_data.csv` | sentence | 400 | SentLen, omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT |
| `zuco_word_raw` | `ZuCo_et_csv_data/word/word_averages_v2.csv` | word | 7129 | nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT, WordLen |
| `sst_full` | `SST_data/combined_full_sst_et.csv` | sentence | 11853 | nFix, GD, TRT, FFD, GPT |
| `sst_train` | `SST_data/train_full_sst.csv` | split | 9482 | nFix, GD, TRT, FFD, GPT |
| `sst_valid` | `SST_data/valid_full_sst.csv` | split | 1185 | nFix, GD, TRT, FFD, GPT |
| `sst_test` | `SST_data/test_full_sst.csv` | split | 1186 | nFix, GD, TRT, FFD, GPT |
| `sst_word_predicted` | `gaze_prediction/data/prediction_test_v2.csv` | word | 191971 | nFix, FFD, GPT, TRT, GD |
| `provo_word` | `gaze_prediction/data/provo.csv` | word | 2659 | nFix, FFD, GPT, TRT, fixProp |

## ZuCo ∩ SST, standard-scaled sentence gaze

400 movie-review sentences that appear in ZuCo Task 1 (normal reading) joined to SST-3 labels. Gaze columns are z-scored after averaging across subjects.

- file: `ZuCo_SST_data/combined_sst_et_standard.csv`
- rows: 400
- columns: sentence_id, sentence, sentiment_label, omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT

- note: This is the table model_ZuCo_SST.py trains on.
- note: Labels: 0=negative, 1=neutral, 2=positive.

### Label mix

| label | name | count | percent |
| ---: | --- | ---: | ---: |
| 0 | negative | 123 | 30.8% |
| 1 | neutral | 137 | 34.2% |
| 2 | positive | 140 | 35.0% |

### Gaze summary

| column | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| omissionRate | -0.000 | 1.001 | -2.448 | -0.025 | 4.245 |
| nFixations | -0.000 | 1.001 | -1.556 | -0.204 | 6.116 |
| meanPupilSize | 0.000 | 1.001 | -1.928 | 0.071 | 2.837 |
| GD | -0.000 | 1.001 | -1.575 | -0.203 | 6.118 |
| TRT | 0.000 | 1.001 | -1.494 | -0.241 | 4.685 |
| FFD | -0.000 | 1.001 | -1.922 | -0.121 | 6.137 |
| SFD | 0.000 | 1.001 | -2.977 | 0.023 | 4.630 |
| GPT | 0.000 | 1.001 | -1.564 | -0.232 | 6.093 |

High correlations (|r| ≥ 0.80):

| left | right | pearson |
| --- | --- | --- |
| nFixations | TRT | 0.958 |
| TRT | GPT | 0.942 |
| nFixations | GPT | 0.912 |

Scaling fingerprint: **standard-like**.

## Subject-averaged ZuCo sentence gaze (raw units)

Per-sentence means across the 12 ZuCo readers, still in milliseconds and fixation counts. No SST labels.

- file: `ZuCo_et_csv_data/average_data.csv`
- rows: 400
- columns: id, SentLen, omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT

- note: Subject 3 contributes only the first 299 rows; later rows average 11 readers.

### Gaze summary

| column | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| SentLen | 17.806 | 8.060 | 3.000 | 17.250 | 42.000 |
| omissionRate | 0.319 | 0.067 | 0.156 | 0.317 | 0.601 |
| nFixations | 1.687 | 0.310 | 1.205 | 1.624 | 3.583 |
| meanPupilSize | 797.009 | 60.942 | 679.653 | 801.328 | 969.674 |
| GD | 141.447 | 21.481 | 107.646 | 137.088 | 272.708 |
| TRT | 202.647 | 47.947 | 131.090 | 191.122 | 427.007 |
| FFD | 116.878 | 7.995 | 101.531 | 115.908 | 165.879 |
| SFD | 71.571 | 10.854 | 39.302 | 71.818 | 121.759 |
| GPT | 241.769 | 56.708 | 153.161 | 228.652 | 586.856 |

High correlations (|r| ≥ 0.80):

| left | right | pearson |
| --- | --- | --- |
| nFixations | TRT | 0.958 |
| TRT | GPT | 0.942 |
| nFixations | GPT | 0.912 |

Scaling fingerprint: **raw or mixed**.

## Subject-averaged ZuCo word gaze (raw units)

7,129 word tokens from the same 400 sentences, averaged over readers.

- file: `ZuCo_et_csv_data/word/word_averages_v2.csv`
- rows: 7129
- columns: id, Sent_ID, Word_ID, Word, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT, WordLen

- note: Sent_ID uses the ZuCo NR suffix, e.g. 0_NR.

### Gaze summary

| column | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| nFixations | 1.111 | 0.711 | 0.000 | 1.000 | 7.250 |
| meanPupilSize | 534.146 | 217.745 | 0.000 | 576.917 | 973.413 |
| GD | 93.769 | 53.086 | 0.000 | 89.083 | 590.083 |
| TRT | 131.207 | 90.933 | 0.000 | 114.083 | 919.500 |
| FFD | 78.633 | 34.836 | 0.000 | 82.364 | 205.750 |
| SFD | 48.257 | 23.797 | 0.000 | 47.636 | 159.750 |
| GPT | 156.634 | 145.419 | 0.000 | 122.917 | 2424.917 |
| WordLen | 4.855 | 2.755 | 0.000 | 4.000 | 21.000 |

High correlations (|r| ≥ 0.80):

| left | right | pearson |
| --- | --- | --- |
| nFixations | TRT | 0.978 |
| GD | TRT | 0.901 |
| meanPupilSize | FFD | 0.895 |
| GD | FFD | 0.895 |
| nFixations | GD | 0.877 |
| meanPupilSize | GD | 0.809 |
| TRT | FFD | 0.805 |

Scaling fingerprint: **raw or mixed**.

## Full SST-3 with sentence-level predicted gaze

11,853 SST sentences with five z-scored gaze channels. Those channels come from aggregating word-level predictions, not from ZuCo readers.

- file: `SST_data/combined_full_sst_et.csv`
- rows: 11853
- columns: sentence_id, sentence, sentiment_label, nFix, GD, TRT, FFD, GPT

- note: This is the table model_full_SST.py trains on.

### Label mix

| label | name | count | percent |
| ---: | --- | ---: | ---: |
| 0 | negative | 4649 | 39.2% |
| 1 | neutral | 2241 | 18.9% |
| 2 | positive | 4963 | 41.9% |

### Gaze summary

| column | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| nFix | 0.000 | 1.000 | -8.550 | -0.071 | 3.455 |
| GD | -0.000 | 1.000 | -12.648 | 0.080 | 1.526 |
| TRT | 0.000 | 1.000 | -7.225 | -0.098 | 4.009 |
| FFD | -0.000 | 1.000 | -8.336 | -0.093 | 3.860 |
| GPT | 0.000 | 1.000 | -8.846 | -0.081 | 3.790 |

High correlations (|r| ≥ 0.80):

| left | right | pearson |
| --- | --- | --- |
| FFD | GPT | 0.999 |
| nFix | GPT | 0.998 |
| nFix | FFD | 0.995 |
| TRT | FFD | 0.995 |
| TRT | GPT | 0.995 |
| nFix | TRT | 0.986 |

Scaling fingerprint: **standard-like**.

## Provo corpus word-level gaze (gaze-prediction source)

2,659 words / 134 sentences from the Provo corpus, used as a natural-reading source for the gaze predictor.

- file: `gaze_prediction/data/provo.csv`
- rows: 2659
- columns: sentence_id, word_id, word, nFix, FFD, GPT, TRT, fixProp

### Gaze summary

| column | mean | std | min | median | max |
| --- | --- | --- | --- | --- | --- |
| nFix | 15.100 | 9.420 | -1.653 | 14.211 | 65.767 |
| FFD | 3.190 | 1.420 | 0.008 | 3.336 | 6.969 |
| GPT | 6.350 | 5.910 | -2.929 | 5.408 | 45.080 |
| TRT | 5.310 | 3.640 | -0.761 | 4.878 | 23.784 |
| fixProp | 67.060 | 26.060 | 3.404 | 71.881 | 106.119 |

High correlations (|r| ≥ 0.80):

| left | right | pearson |
| --- | --- | --- |
| nFix | TRT | 0.982 |
| FFD | fixProp | 0.960 |
| nFix | fixProp | 0.918 |
| FFD | TRT | 0.915 |
| nFix | FFD | 0.899 |
| GPT | TRT | 0.896 |
| TRT | fixProp | 0.885 |
| nFix | GPT | 0.867 |
| FFD | GPT | 0.800 |

Scaling fingerprint: **raw or mixed**.

## Split audits

### ZuCo ∩ SST 80/10/10 (ok)

- train/valid/test rows: 320 / 40 / 40
- unique union IDs: 400
- disjoint splits: True
- covers parent table: True
- label mix (negative / neutral / positive):
  - parent: 30.8%, 34.2%, 35.0%
  - train: 32.2%, 33.4%, 34.4%
  - valid: 17.5%, 35.0%, 47.5%
  - test: 32.5%, 40.0%, 27.5%

### Full SST 80/10/10 (ok)

- train/valid/test rows: 9482 / 1185 / 1186
- unique union IDs: 11853
- disjoint splits: True
- covers parent table: True
- label mix (negative / neutral / positive):
  - parent: 39.2%, 18.9%, 41.9%
  - train: 39.1%, 19.3%, 41.5%
  - valid: 40.2%, 17.6%, 42.2%
  - test: 39.0%, 16.8%, 44.2%
