# Glossary

Terms as they are used in **this** repo, not as a general
psycholinguistics textbook.

| Term | Meaning here |
| --- | --- |
| **ZuCo** | Zurich Cognitive Language Processing Corpus. Task 1 is the movie-review / sentiment reading task whose sentences overlap SST. |
| **Task 1 / 2 / 3** | ZuCo reading paradigms. This checkout’s CSVs are Task 1 (`*_SR.csv`, `Sent_ID` suffix `_NR`). `DataTransformer` still knows about tasks 2 and 3 for MATLAB conversion. |
| **SST** | Stanford Sentiment Treebank. Full set ≈ 11.8k sentences in `SST_data/`. |
| **ZuCo ∩ SST** | The 400 sentences that appear in both: real gaze + SST-style labels. |
| **Measured gaze** | Eye-tracking recorded from ZuCo participants, then averaged. |
| **Transferred gaze** | Gaze columns on full SST, produced by a predictor / mapping, not by an eye tracker. |
| **Late fusion** | Concatenate a text vector and a gaze vector, then classify. No cross-attention. |
| **`pooler_output`** | Hugging Face’s pooled first-token state (768-d for base models). |
| **`model_type`** | Header string: `bert`, `roberta`, `bert_eye_tracking`, `roberta_eye_tracking`. |
| **nFix / nFixations** | Fixation count (per word, or mean over fixated words). |
| **FFD** | First Fixation Duration. |
| **GD** | Gaze Duration (first-pass time). |
| **TRT** | Total Reading Time (all passes). |
| **GPT** | Go-Past Time (regression-path duration). |
| **SFD** | Single Fixation Duration (only when there is exactly one fixation). |
| **omissionRate** | Fraction of words in the sentence with no fixation. |
| **meanPupilSize** | Mean pupil size over fixated words. Unused by the fusion head. |
| **WordLen / SentLen** | Character length of a token / number of words in a sentence. |
| **Standard scaling** | z-score: `(x - mean) / std`. Used by `model_ZuCo_SST.py`. |
| **Min–max scaling** | `(x - min) / (max - min)`. Sibling CSV, unused by the default trainer. |
| **0–100 scaling** | `convert_zuco_data.py` helper for the prediction schema. |
| **NR** | “Normal reading” suffix on `Sent_ID` (`12_NR`). |
| **TSR** | Task-3-style suffix in `DataTransformer` word mode (`_TSR`). Not in the checked-in word averages. |
| **SR** | Filename stem `{subject}_SR.csv` for sentence-level exports. |
| **Subject index** | `DataTransformer` uses `0..11`. Files are `1_SR.csv` … `12_SR.csv`. File `3_SR` is subject index `2`. |
| **Label 0 / 1 / 2** | Negative / Neutral / Positive. |
| **Weighted F1** | sklearn `average='weighted'` — support-weighted mean of per-class F1. |
| **KFold here** | Always `StratifiedKFold`, 5 splits, `random_state=42`, on the 400. |
| **Placeholder zeros** | `convert_sst_to_et.py` output. Not a real reading. |
| **PROVO** | External corpus format used as `gaze_prediction/data/provo.csv`. Last column `fixProp`. |
| **CLS** | First encoder token. BERT uses `[CLS]`; RoBERTa uses `<s>`. Both expose `pooler_output`. |
| **B, T, H** | Batch, sequence length (128), hidden size (768). Gaze is `(B, 5)`. |

When a CSV header disagrees with this table, the header wins for
that file; see the naming chart in
[03-eye-tracking-features.md](03-eye-tracking-features.md).
