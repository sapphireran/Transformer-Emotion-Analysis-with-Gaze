# References and related personal work

Short reading list for the datasets and the fusion idea. These are not
citations of company documents.

## Datasets

- **ZuCo** — Zurich Cognitive Language Processing Corpus. Simultaneous EEG and
  eye-tracking while people read sentences. Task 1 is normal reading of movie
  reviews (the slice this repo uses). Later tasks add sentiment annotation and
  relation-extraction reading.
  Hollenstein et al., *Proceedings of the 11th LREC* (2018), plus the ZuCo 2.0
  follow-up.
  <https://osf.io/q3zws/>

- **Stanford Sentiment Treebank (SST)** — Socher et al., EMNLP 2013. Fine-
  grained phrase-level polarity; this repo uses sentence-level 3-class labels
  (`NEGATIVE` / `NEUTRAL` / `POSITIVE` → 0 / 1 / 2).

- **PROVO** — Luke & Christianson, *Behavior Research Methods* (2018).
  Word-level eye movements on passages. `gaze_prediction/data/provo.csv` is a
  converted extract (`nFix`, `FFD`, `GPT`, `TRT`, `fixProp`).

## Models the scripts wrap

- Devlin et al., *BERT*, NAACL 2019. `bert-base-uncased`.
- Liu et al., *RoBERTa*, 2019. `roberta-base`.
- Hugging Face `transformers` sequence-classification heads for the text-only
  ablations.

## Gaze + NLP (context, not dependencies)

Work that mixes reading-time features with language models, in the same
neighbourhood as this personal project:

- Hollenstein, Barrett, et al. on ZuCo-based NLP probes (sentiment, relation
  extraction, cognitive-feature augmentation).
- Barrett, Bingel, Hollenstein, et al. on using eye-tracking as weak
  supervision / attention priors.
- Token-level reading-time prediction papers that motivate
  `gaze_prediction/data/prediction_test_v2.csv` (the predictor itself is not
  in this repo).

This repository's fusion is simpler than those lines of work: a 5→16 linear
map concatenated with the encoder pooler.

## Eye-tracking measures

Standard first-pass / regression-path definitions (Rayner; Clifton, Staub &
Rayner) are what ZuCo's `FFD`, `GD`, `GPT`, `TRT`, `SFD` follow. See
[features.md](features.md) for how this repo aggregates them to sentence
level.

## Other personal repos in the same cluster

Public personal projects on the same GitHub account that share the
multimodal / sentiment theme (separate trees, not imported here):

- `Transformer-based-Multimodal-Sentiment-Analysis`
- `Twitter-Sarcasm-Detection-Multimodal_with_Emoji` (UCPH CCS2, 2023)
- `Danish-News-Summarization` (ITU NLP/DL, 2023)

Those are listed so a future clone of *this* repo can find the sibling
experiments. Nothing in `examples/` calls them.
