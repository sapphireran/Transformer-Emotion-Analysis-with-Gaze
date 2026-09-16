# Citations and data sources

This repository is a personal experiment. If you reuse the **ideas** or the **checked-in derived tables**, cite the original corpora. The CSVs here are convenience exports, not a replacement for the official releases.

## ZuCo

Hollenstein, N., Rotsztejn, J., Troendle, M., Pedroni, A., Zhang, C., & Langer, N. (2018).
**ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading.**
*Scientific Data.*

- Task used here: sentiment reading (task 1 / SR), 400 sentences, 12 subjects.
- Official page and download instructions are on the ZuCo project site (University of Zurich / ETH-associated releases).
- MATLAB field names (`nFixations`, `FFD`, `GPT`, `TRT`, `GD`, `SFD`, `omissionRate`) follow that release. `utils_ZuCo.DataTransformer` is an adapter, not a redefinition.

## Stanford Sentiment Treebank

Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., & Potts, C. (2013).
**Recursive deep models for semantic compositionality over a sentiment treebank.**
*EMNLP.*

- This repo uses a **sentence-level** three-class projection (`NEGATIVE` / `NEUTRAL` / `POSITIVE` → 0 / 1 / 2), not the original five-way fine-grained phrase labels.
- `SST_data/stts_all_sentence_level.csv` is an export used as a conversion input.

## Provo

Luke, S. G., & Christianson, K. (2018).
**The Provo Corpus: A large eye-tracking corpus with predictability norms.**
*Behavior Research Methods.*

- Only a small derived table is checked in: `gaze_prediction/data/provo.csv` (2,659 word rows) plus `result/provo_data_scatter_hist_plots.png`.

## Encoders

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). **BERT.** *NAACL.*

Liu, Y., et al. (2019). **RoBERTa: A robustly optimized BERT pretraining approach.** arXiv:1907.11692.

Weights are pulled at runtime via `transformers` (`bert-base-uncased`, `roberta-base`). They are not stored in this git tree.

## What this repo does *not* claim

- It does not redistribute the official ZuCo `.mat` recordings.
- It does not claim that predicted SST gaze is a substitute for a new eye-tracking study.
- Plot titles and script comments were written for personal use; they are not a publication-ready methods section.

When in doubt, download the corpus from its authors and re-run Stage 1 of [data-pipeline.md](data-pipeline.md).
