# Dataset inspection

| name            | rows  | cols | role                            |
|-----------------|-------|------|---------------------------------|
| zuco_text       | 400   | 3    | ZuCo text + labels              |
| zuco_standard   | 400   | 11   | ZuCo text + z-scored gaze       |
| zuco_minmax     | 400   | 11   | ZuCo text + min-max gaze        |
| zuco_train      | 320   | 11   | ZuCo 80% split                  |
| zuco_valid      | 40    | 11   | ZuCo 10% valid                  |
| zuco_test       | 40    | 11   | ZuCo 10% test                   |
| zuco_subject_1  | 400   | 10   | ZuCo subject 1 sentences        |
| zuco_average    | 400   | 10   | ZuCo subject-mean sentences     |
| zuco_word_avg   | 7129  | 12   | ZuCo word-level subject mean    |
| sst_combined    | 11853 | 8    | Full SST + projected gaze       |
| sst_train       | 9482  | 8    | Full SST train                  |
| sst_valid       | 1185  | 8    | Full SST valid                  |
| sst_test        | 1186  | 8    | Full SST test                   |
| pred_word_small | 1751  | 8    | Word-level predicted ET (small) |

### zuco_text

- file: `ZuCo_SST_data/ssts_ZuCo.csv`
- columns: sentence_id, sentence, sentiment_label
- sample: Presents a good case while failing to provide a reason for us to care beyond the very basic d...

### zuco_standard

- file: `ZuCo_SST_data/combined_sst_et_standard.csv`
- columns: sentence_id, sentence, sentiment_label, omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
- sample: Presents a good case while failing to provide a reason for us to care beyond the very basic d...

### zuco_minmax

- file: `ZuCo_SST_data/combined_sst_et_min_max.csv`
- columns: sentence_id, sentence, sentiment_label, omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
- sample: Presents a good case while failing to provide a reason for us to care beyond the very basic d...

### zuco_train

- file: `ZuCo_SST_data/train.csv`
- columns: sentence_id, sentence, sentiment_label, omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
- sample: Slow, silly and unintentionally hilarious.

### zuco_valid

- file: `ZuCo_SST_data/valid.csv`
- columns: sentence_id, sentence, sentiment_label, omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
- sample: It will guarantee to have you leaving the theater with a smile on your face.

### zuco_test

- file: `ZuCo_SST_data/test.csv`
- columns: sentence_id, sentence, sentiment_label, omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
- sample: It's a head-turner -- thoughtfully written, beautifully read and, finally, deeply humanizing.

### zuco_subject_1

- file: `ZuCo_et_csv_data/1_SR.csv`
- columns: id, SentLen, omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
- sample: (none)

### zuco_average

- file: `ZuCo_et_csv_data/average_data.csv`
- columns: id, SentLen, omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
- sample: (none)

### zuco_word_avg

- file: `ZuCo_et_csv_data/word/word_averages_v2.csv`
- columns: id, Sent_ID, Word_ID, Word, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT, WordLen
- sample: presents

### sst_combined

- file: `SST_data/combined_full_sst_et.csv`
- columns: sentence_id, sentence, sentiment_label, nFix, GD, TRT, FFD, GPT
- sample: The Rock is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a...

### sst_train

- file: `SST_data/train_full_sst.csv`
- columns: sentence_id, sentence, sentiment_label, nFix, GD, TRT, FFD, GPT
- sample: Plays like a volatile and overlong W magazine fashion spread .

### sst_valid

- file: `SST_data/valid_full_sst.csv`
- columns: sentence_id, sentence, sentiment_label, nFix, GD, TRT, FFD, GPT
- sample: `` An entire film about researchers quietly reading dusty old letters . ''

### sst_test

- file: `SST_data/test_full_sst.csv`
- columns: sentence_id, sentence, sentiment_label, nFix, GD, TRT, FFD, GPT
- sample: There 's nothing remotely topical or sexy here .

### pred_word_small

- file: `gaze_prediction/data/prediction_test.csv`
- columns: sentence_id, word_id, word, nFix, FFD, GPT, TRT, GD
- sample: samuel
