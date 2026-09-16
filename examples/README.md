# Examples

Personal walkthroughs for the CSVs and the concat-fusion idea. None of these
download BERT/RoBERTa. None of them touch workplace code.

From the repository root:

```bash
export PYTHONPATH=.
python examples/01_inspect_datasets.py
python examples/02_label_and_feature_stats.py
python examples/03_compare_scaling.py
python examples/04_toy_fusion_forward.py
python examples/05_gaze_only_baseline.py
python examples/06_word_to_sentence.py
python examples/07_schema_and_glossary.py
python examples/08_split_sanity.py
```

| Script | What it prints |
| --- | --- |
| `01_inspect_datasets.py` | Row counts, columns, paths for every trainer table |
| `02_label_and_feature_stats.py` | Class shares + gaze–label correlations |
| `03_compare_scaling.py` | Raw vs z-score vs min-max on the reader-mean file |
| `04_toy_fusion_forward.py` | 5→16 concat geometry and a tiny hashed-BoW fusion fit |
| `05_gaze_only_baseline.py` | Majority vs logistic-on-gaze, 5-fold, both corpora |
| `06_word_to_sentence.py` | Re-aggregate `word_averages_v2.csv` and compare to sentence ET |
| `07_schema_and_glossary.py` | Feature glossary + which columns each trainer actually reads |
| `08_split_sanity.py` | Confirm 80/10/10 disjoint ids and matching class mix |

Helpers live in `tea_gaze/`. The historical `model_*_SST.py` files are not
imported here on purpose.
