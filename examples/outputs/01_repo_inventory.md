# Repository inventory

Personal checklist of tables this clone can consume without MATLAB.

## Experiment tracks

| track | sentences | script | gaze |
| --- | --- | --- | --- |
| ZuCo–SST overlap | 400 | model_ZuCo_SST.py | Human eye-tracking, 12 readers, sentence-averaged then z-scored |
| Full SST with projected gaze | 11853 | model_full_SST.py | Projected / predicted features, not a 12-reader recording of the full SST |

## Consumer tables

| path | rows | schema | note |
| --- | --- | --- | --- |
| ZuCo_SST_data/combined_sst_et_standard.csv | 400 | yes | 5-fold training table |
| ZuCo_SST_data/combined_sst_et_min_max.csv | 400 | yes | same 400 sentences, min-max gaze |
| ZuCo_SST_data/ssts_ZuCo.csv | 400 | yes | text + label only |
| ZuCo_SST_data/train.csv | 320 | yes | 320-row leftover split |
| ZuCo_SST_data/valid.csv | 40 | yes | 40-row leftover split |
| ZuCo_SST_data/test.csv | 40 | yes | 40-row leftover split |
| SST_data/combined_full_sst_et.csv | 11853 | yes | projected-gaze SST |
| SST_data/train_full_sst.csv | 9482 | yes | 80% |
| SST_data/valid_full_sst.csv | 1185 | yes | 10% |
| SST_data/test_full_sst.csv | 1186 | yes | 10% |
| ZuCo_et_csv_data/average_data.csv | 400 | yes | index-mean of 12 readers |
| ZuCo_et_csv_data/standard_scaled_average_data.csv | 400 | yes | z-scored average_data |
| ZuCo_et_csv_data/word/word_averages_v2.csv | 7129 | yes | word-level means |
| gaze_prediction/data/provo.csv | 2659 | yes | PROVO, fixProp not GD |
| gaze_prediction/data/prediction_test.csv | 1751 | yes | predicted word gaze |

## Twelve readers

| path | rows | schema | note |
| --- | --- | --- | --- |
| ZuCo_et_csv_data/1_SR.csv | 400 | yes | 400 raw-ish sentence rows |
| ZuCo_et_csv_data/2_SR.csv | 400 | yes | 400 raw-ish sentence rows |
| ZuCo_et_csv_data/3_SR.csv | 299 | yes | 299 rows and remapped ids |
| ZuCo_et_csv_data/4_SR.csv | 400 | yes | 400 raw-ish sentence rows |
| ZuCo_et_csv_data/5_SR.csv | 400 | yes | 400 raw-ish sentence rows |
| ZuCo_et_csv_data/6_SR.csv | 400 | yes | 400 raw-ish sentence rows |
| ZuCo_et_csv_data/7_SR.csv | 400 | yes | 400 raw-ish sentence rows |
| ZuCo_et_csv_data/8_SR.csv | 400 | yes | 400 raw-ish sentence rows |
| ZuCo_et_csv_data/9_SR.csv | 400 | yes | 400 raw-ish sentence rows |
| ZuCo_et_csv_data/10_SR.csv | 400 | yes | 400 raw-ish sentence rows |
| ZuCo_et_csv_data/11_SR.csv | 400 | yes | 400 raw-ish sentence rows |
| ZuCo_et_csv_data/12_SR.csv | 400 | yes | 400 raw-ish sentence rows |

## File notes

| path | kind | rows | note |
| --- | --- | --- | --- |
| ZuCo_SST_data/combined_sst_et_standard.csv | train table | 400 | Z-scored gaze + labels. model_ZuCo_SST.py reads this. |
| ZuCo_SST_data/combined_sst_et_min_max.csv | train table | 400 | Same sentences, min-max gaze. Not used by the training scripts. |
| ZuCo_SST_data/ssts_ZuCo.csv | text + label | 400 | No gaze columns. Producer leftover from convert_full_SST.py. |
| ZuCo_SST_data/train.csv | split | 320 | Random 80% split. Valid is not stratified (7/14/19). |
| ZuCo_SST_data/valid.csv | split | 40 | Too small for a stable accuracy number. |
| ZuCo_SST_data/test.csv | split | 40 | Disjoint from train/valid; unused by 5-fold script. |
| ZuCo_et_csv_data/{1-12}_SR.csv | per reader | 400 (299 for #3) | Closer to raw ms / pupil. Subject 3 is remapped after id 149. |
| ZuCo_et_csv_data/average_data.csv | mean over readers | 400 | Index-averaged; contaminated for ids 150–398 by subject 3. |
| ZuCo_et_csv_data/word/word_averages_v2.csv | word means | 7129 | 400 sentences, Sent_ID like 0_NR. Same index-average issue. |
| SST_data/combined_full_sst_et.csv | train table | 11853 | Z-scored projected gaze. model_full_SST.py uses the split files. |
| SST_data/stts_all_sentence_level.csv | raw SST | 11853 | sentence,POS/NEG/NEU — no header. |
| gaze_prediction/data/provo.csv | external ET | 2659 | PROVO word table with fixProp instead of GD. |
| gaze_prediction/data/prediction_test.csv | predicted ET | 1751 | 100 sentences of model-predicted word gaze. |
