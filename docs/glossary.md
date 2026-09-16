# Glossary

| term | meaning in this clone |
| --- | --- |
| ZuCo | Zurich Cognitive Language Processing Corpus (EEG + ET). Task 1 normal reading is what the CSVs came from. |
| SST | Stanford Sentiment Treebank movie reviews, collapsed here to three labels. |
| NR | Normal reading. Word `Sent_ID` values look like `12_NR`. |
| TSR | Task 3 sentiment reading in ZuCo; not the checked-in sentence tables. |
| FFD | First Fixation Duration. |
| SFD | Single Fixation Duration. |
| GD | Gaze Duration / first-pass time. |
| GPT | Go-Past Time. |
| TRT | Total Reading Time. |
| nFixations / nFix | Fixation count. |
| omissionRate | Fraction of words never fixated. |
| pooler | 768-d sentence vector from BERT/RoBERTa (`pooler_output`). |
| late fusion | Concatenate text vector and a projected gaze vector, then classify. |
| projected gaze | Features on full SST that are not twelve-reader recordings. |
| compact id | Row `id` in `3_SR.csv` after skipped sentences were dropped. |
| original id | Sentence index 0..399 used by the other eleven readers and by the labeled tables. |
| index mean | `groupby(level=0)` average — what `average_data.csv` is. |
| remapped mean | Average after sending reader 3's compact ids back to 250..398. |
| 5-fold script | `model_ZuCo_SST.py`. |
| split script | `model_full_SST.py`. |
| personal layer | `zuco_lab/`, `examples/`, `docs/` — read-only on the original tables. |
