# Citations (public datasets)

This personal project only uses public research corpora. Nothing here is
workplace data.

## ZuCo

Hollenstein, N., Rotsztejn, J., Troendle, M., Pedroni, A., Zhang, C., &
Langer, N. (2018). *ZuCo, a simultaneous EEG and eye-tracking resource for
natural sentence reading.* Scientific Data.

The MATLAB layout that `utils_ZuCo.py` walks (`sentenceData`, per-word
`nFixations`, `FFD`, `GD`, `TRT`, `GPT`, `SFD`, `meanPupilSize`,
`omissionRate`) is from that release. This repo keeps derived CSVs only.

ZuCo 2.0 exists; these scripts are written against the original task1 / task2
/ task3 subject files (12 readers).

## Stanford Sentiment Treebank

Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., &
Potts, C. (2013). *Recursive Deep Models for Semantic Compositionality Over
a Sentiment Treebank.* EMNLP.

`SST_data/stts_all_sentence_level.csv` is a sentence-level export with the
three-way labels this project trains on. Fine-grained 5-class SST phrases
are not used.

## PROVO

Luke, S. G., & Christianson, K. (2018). *The Provo Corpus: A large
eye-tracking corpus with predictability ratings.* Behavior Research Methods.

`gaze_prediction/data/provo.csv` is a small extract used as a predicted-gaze
relative, not as a trainer input.

## Transformer checkpoints

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training
of Deep Bidirectional Transformers for Language Understanding.* NAACL.

Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining
Approach.* arXiv:1907.11692.

The scripts call `from_pretrained('bert-base-uncased')` and
`from_pretrained('roberta-base')`.

## Related reading I keep next to this repo

Hollenstein, N. & Zhang, C. work on evaluating word embeddings / NLP models
with cognitive data (eye tracking, EEG). The concat-fusion idea in
`EyeTrackingModel` is a small personal take on that line: attach summary
reading measures to a sentence encoder, ask whether sentiment F1 moves.

If I write this up later, the claim has to stay no stronger than the
baselines in `examples/05_gaze_only_baseline.py` plus a seeded GPU comparison
of `roberta` vs `roberta_eye_tracking`.
