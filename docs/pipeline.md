# Pipeline

How a movie-review sentence becomes a 784-d fused vector. Several
producer folders are missing from this clone; the consumer CSVs are
not.

## Intended ZuCo path

```
ZuCo .mat (12 readers, task 1 NR)
        │  utils_ZuCo.DataTransformer
        │  read_ZuCo_mat.py
        ▼
ZuCo_et_csv_data/{1-12}_SR.csv          (and word/{1-12}_SR.csv)
        │  get_average_sentence_level.py
        ▼
average_data.csv
standard_scaled_average_data.csv
min_max_scaled_average_data.csv
        │  join on sentence id
        │  labels from SST-overlap text
        ▼
ZuCo_SST_data/combined_sst_et_standard.csv
        │  model_ZuCo_SST.py
        ▼
5-fold BERT / RoBERTa ± gaze
```

`read_ZuCo_mat.py` still writes `et_csv_data/`. The checked-in
directory is `ZuCo_et_csv_data/`. `utils_ZuCo.get_matfiles` uses
Windows backslashes and expects `ZuCo_mat_data/`, which is absent.

Subject-index quirks in `DataTransformer` (task 1 subject 2, plus
other task/subject holes) are why file 3 has 299 rows. See
[subject alignment](subject-alignment.md).

## Intended SST-overlap labels

```
ZuCo_SST_data/all/{NEGATIVE,POSITIVE,NEUTRAL}/*.txt
        │  convert_full_SST.py  or  ZuCo_SST_data/save_SST_data.py
        ▼
ssts_ZuCo.csv   (sentence_id, sentence, sentiment_label)
        │  join gaze on sentence_id
        ▼
combined_sst_et_{standard,min_max}.csv
        │  ZuCo_SST_data/spilt.py   (typo)
        ▼
train.csv / valid.csv / test.csv
```

`all/` is not checked in. `ssts_ZuCo.csv` is. Label mapping is
`NEGATIVE=0`, `NEUTRAL=1`, `POSITIVE=2`.

## Intended full-SST path

```
SST_data/stts_all_sentence_level.csv     (no header)
        │  tokenize, attach projected gaze
        ▼
combined_full_sst_et.csv
        │  SST_data/spilt.py
        ▼
train_full_sst.csv / valid_full_sst.csv / test_full_sst.csv
        │  model_full_SST.py
        ▼
models/best_{model_type}_model.pth
        │  test loop  ← keeps last batch only
        ▼
printed test Acc / P / R / F1
```

`convert_sst_to_et.py` writes a word table with **zeros** for every
gaze column (`sst_et_test.csv`). That is a tokenizer scaffold, not
measurements.

## What the training scripts actually read

| script | path | gaze columns |
| --- | --- | --- |
| `model_ZuCo_SST.py` | `ZuCo_SST_data/combined_sst_et_standard.csv` | `nFixations, FFD, GPT, TRT, GD` |
| `model_full_SST.py` | `SST_data/{train,valid,test}_full_sst.csv` | `nFix, FFD, GPT, TRT, GD` |

Both tokenize with `max_length=128`, `padding='max_length'`. Both
build a Hugging Face `Dataset` just to get `input_ids` /
`attention_mask`, then jump back to pandas.

## Tokenization vs ZuCo words

ZuCo word tables are cleaned with `re.sub('[^\w\s]', '', word.content)`
and lowercased only at the start of the sentence. BERT/RoBERTa
tokenizers do WordPiece / BPE on the raw SST string. There is no
alignment between word-level ET and subword ids in this clone. Fusion
is **sentence-level only**: five scalars for the whole sentence.

## Gaze-prediction side path

```
word_averages_v2.csv
        │  gaze_prediction/data/convert_zuco_data.py
        ▼
scaled word table (nFix, FFD, GPT, TRT, GD)
        │  some external predictor (not in this clone)
        ▼
prediction_test.csv
provo.csv          (public corpus, different columns)
```

Not wired into either training script.
