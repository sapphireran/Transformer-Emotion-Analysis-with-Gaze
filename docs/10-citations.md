# Citations and scope

Personal notes for a public research checkout. Not a paper. Not workplace
code.

## Data

- Hollenstein, N., Rotsztejn, J., Troendle, M., Pedroni, A., Zhang, C.,
  & Langer, N. (2018). ZuCo, a simultaneous EEG and eye-tracking
  resource for natural sentence reading. *Scientific Data*.
- Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A.,
  & Potts, C. (2013). Recursive deep models for semantic compositionality
  over a sentiment treebank. *EMNLP*.
- Luke, S. G., & Christianson, K. (2018). The Provo Corpus: A large
  eye-tracking corpus with predictability ratings. *Behavior Research
  Methods*. Used here only as the committed `provo.csv` schema
  (`fixProp` instead of `GD`).

## Models the original trainers download

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT.
- Liu, Y., et al. (2019). RoBERTa.

Examples do not download those weights.

## What this notes layer does not claim

- It does not re-run `model_ZuCo_SST.py` or `model_full_SST.py`.
- It does not treat ridge+hash numbers as paper baselines.
- It does not rewrite `average_data.csv` after documenting the reader-3
  shift.
- It does not include MATLAB sources, EEG, or company data.
