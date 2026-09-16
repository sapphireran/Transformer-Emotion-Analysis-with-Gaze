# Glossary

| Term | Meaning in this repo |
| --- | --- |
| SST / SST-3 | Stanford Sentiment Treebank movie snippets, collapsed to negative / neutral / positive |
| ZuCo | Zurich Cognitive Language Processing Corpus (EEG + eye-tracking). Task 1 = normal reading of reviews |
| NR | Normal reading condition. Word `Sent_ID` values end in `_NR` |
| SR | Sentiment reading — used in per-subject filenames `{1-12}_SR.csv` |
| FFD | First fixation duration |
| GD | Gaze duration (first-pass time) |
| TRT | Total reading time |
| GPT | Go-past / regression-path time |
| SFD | Single-fixation duration |
| nFix / nFixations | Fixation count on the region |
| omissionRate | Fraction of words with no fixation |
| pooler_output | 768-d vector from BERT/RoBERTa used as the sentence embedding |
| late fusion | Concatenate text embedding and projected gaze, then classify |
| predicted gaze | Word reading measures from a model, not from ZuCo readers |
| Provo | Word-level reading corpus used as a gaze-prediction source |
| Track A | 400 ZuCo ∩ SST sentences with human gaze |
| Track B | 11,853 SST sentences with predicted gaze |
| weighted F1 | sklearn F1 weighted by support — what `model_*.py` prints |
| macro F1 | Unweighted mean of per-class F1 — what the examples also print |
