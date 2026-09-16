# Citations and data terms

Personal research notes. Confirm the current license on each corpus before
any public redistribution of **derived** CSVs; this repo already contains
processed gaze and SST text.

## Datasets

**ZuCo (Zurich Cognitive Language Processing Corpus)**  
Hollenstein, N., Rotsztejn, J., Troendle, M., Pedroni, A., Zhang, C., &
Jäger, L. (2018). *ZuCo, a simultaneous EEG and eye-tracking resource for
natural sentence reading.* Scientific Data.  
Task 1 in this code is **normal reading** of movie-review sentences
(`Sent_ID` suffix `_NR`). EEG is recorded in ZuCo but **not** used here.

**Stanford Sentiment Treebank (SST)**  
Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., &
Potts, C. (2013). *Recursive Deep Models for Semantic Compositionality Over
a Sentiment Treebank.* EMNLP.  
Labels in the joined tables are collapsed to `{negative, neutral, positive}`
as 0 / 1 / 2.

**PROVO**  
Luke, S. G., & Christianson, K. — Provo Corpus eye-tracking norms. The
file `gaze_prediction/data/provo.csv` is a **predicted / reformatted**
slice (columns `nFix, FFD, GPT, TRT, fixProp`), not a drop-in replacement
for the official release.

## Models

**BERT**  
Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT:
Pre-training of Deep Bidirectional Transformers for Language Understanding.*
NAACL. Weights: `bert-base-uncased`.

**RoBERTa**  
Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining
Approach.* arXiv:1907.11692. Weights: `roberta-base`.

## Related reading (gaze + NLP)

Not dependencies of the code, but the usual context for “does eye-tracking
help sentiment / complexity models”:

- Hollenstein, Barrett, et al. — work on ZuCo for NLP probing and gaze
  prediction.
- Barrett, Bingel, Hollenstein, et al. — combining gaze with RNNs for
  sequence labeling.
- Mathias, Klerke, et al. — cognitive features as auxiliary signals.

## How this repo uses the names

| Name in docs | Meaning here |
| --- | --- |
| Track A | Human ZuCo gaze on ~400 SST sentences |
| Track B | Predicted gaze on full SST |
| Fusion | Concatenate `pooler_output` with `Linear(5, 16)(gaze)` |
| NR | Normal reading (ZuCo task 1 / 2 suffix in word tables) |
| TSR | Task-specific / sentiment reading suffix in `DataTransformer` for task 3 |

## Hugging Face weights

Training scripts call `from_pretrained('bert-base-uncased')` and
`'roberta-base'` with no local cache pin. Record the `transformers` version
in any personal result log; Hub defaults move.
