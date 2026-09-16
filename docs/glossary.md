# Glossary

| Term | Meaning in this repo |
| --- | --- |
| **SST** | Stanford Sentiment Treebank. Movie-review sentences with polarity labels. |
| **ZuCo** | Zurich Cognitive Language Processing Corpus. Simultaneous EEG + eye tracking during reading. Task 1 used SST sentences. |
| **PROVO** | Provo Corpus — a separate eye-tracking corpus with predictability norms. Used here as a *shape* reference for predicted gaze, not as training labels. |
| **Gaze / ET** | Eye-tracking features, not a camera gaze-angle model. |
| **nFix / nFixations** | Fixation count. |
| **FFD** | First fixation duration. |
| **GD** | Gaze duration (first-pass time). |
| **TRT** | Total reading time. |
| **GPT** | Go-past / regression-path time. |
| **SFD** | Single fixation duration. |
| **omissionRate** | Fraction of words never fixated. |
| **pooler_output** | 768-d vector from BERT/RoBERTa used as the sentence embedding. |
| **Fusion** | Concatenate `pooler_output` with a 16-d linear projection of the five gaze features, then classify. |
| **Full SST track** | 11,853 sentences, predicted sentence-level gaze, 80/10/10 holdout. |
| **ZuCo track** | 400 sentences, measured subject-averaged gaze, 5-fold CV. |
| **NR** | Suffix on ZuCo `Sent_ID` (`0_NR`) for the sentiment / normal-reading task. |
| **TSR** | Task-3 style `Sent_ID` suffix in `DataTransformer`; not used in the checked-in CSVs. |
| **Weighted F1** | `sklearn` F1 with `average='weighted'` — class F1s weighted by support. |
| **Predicted gaze** | Model-generated reading-time features. Not human measurements. |
| **Skip** | Word-level row with `nFixations == 0`. |
| **`spilt.py`** | The split scripts. Typo, not a third pipeline. |
