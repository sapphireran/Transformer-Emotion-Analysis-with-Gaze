# Citations and data sources

Personal bibliography for this repo. These are the public corpora and
model checkpoints the scripts assume. License terms belong to the
original publishers; this checkout only stores derived CSVs.

## Corpora

### Zurich Cognitive Language Processing Corpus (ZuCo)

Hollenstein, N., Rotsztejn, J., Troendle, M., Pedroni, A., Zhang, C.,
& Langer, N. (2018). *ZuCo, a simultaneous EEG and eye-tracking
resource for natural sentence reading.* Scientific Data.

Used here: Task 1 sentiment-reading (`SR`) recordings from twelve
readers. Sentence- and word-level tables in `ZuCo_et_csv_data/` are
exports from the public MATLAB release via `utils_ZuCo.DataTransformer`.

Typical public landing page: the ZuCo project page at the University of
Zurich / OSF. MATLAB files themselves are **not** in this git repo.

### Stanford Sentiment Treebank (SST)

Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A.,
& Potts, C. (2013). *Recursive deep models for semantic compositionality
over a sentiment treebank.* EMNLP.

Used here: sentence-level movie-review strings with a three-way label
(negative / neutral / positive). `SST_data/stts_all_sentence_level.csv`
is the text dump; `ZuCo_SST_data/ssts_ZuCo.csv` is the ZuCo-aligned
subset.

### PROVO

Luke, S. G., & Christianson, K. (2018). *The Provo Corpus: A large
eye-tracking corpus with predictability norms.* Behavior Research
Methods.

`gaze_prediction/data/provo.csv` is a compact word-level extract
(`nFix, FFD, GPT, TRT, fixProp`). It is a reference table for the
prediction work, not an input to `model_*.py`.

## Model checkpoints

Training scripts call Hugging Face ids:

| Id | Role |
| --- | --- |
| `bert-base-uncased` | text-only and fusion encoder |
| `roberta-base` | text-only and fusion encoder (default `model_type`) |

Tokenizers match the encoder. First run downloads weights if they are
not already cached.

## Measures

Reading-time definitions follow the usual psycholinguistic first-pass /
late split (Rayner, 1998, *Eye movements in reading and information
processing*; Clifton, Staub, & Rayner, 2007). See
[gaze-features.md](gaze-features.md) for how this repo aggregates them.

## Fusion setup

Late fusion of a 768-d pooler vector with a linear map of five sentence
gaze features is the personal experiment. It is **not** a reimplementation
of a specific published architecture. Related lines of work, if you want
reading:

- Hollenstein & Zhang, *Entity recognition with eye-tracking features*
- Barrett, Bingel, Hollenstein, et al., *Sequence classification with
  human attention*
- SST + cognitive-feature papers that append reading times to a
  sentence encoder

Cite those if you write up a comparison. Do not cite this repo as if it
were one of them.

## What you can redistribute

The original ZuCo and SST releases have their own licenses. The scripts
and documentation in this personal repo can be copied with the repo.
The derived CSVs are convenient for reruns; if you publish a dataset,
point at the official ZuCo / SST sources rather than treating these
CSVs as a new corpus.
