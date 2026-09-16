# PROVO vs predicted word gaze

These tables live under ``gaze_prediction/data/`` and are not read by the two training scripts. They are the leftover of a word-level prediction experiment (PROVO as a public ET corpus, plus a predicted file whose sentence ids start at 300).

## Coverage

| table | rows | sentences | first sentence_id | last sentence_id | extra column |
| --- | --- | --- | --- | --- | --- |
| provo.csv | 2659 | 134 | 0 | 133 | fixProp |
| prediction_test.csv | 1751 | 100 | 300 | 399 | GD |

## PROVO feature ranges

| feature | mean | std | min | max |
| --- | --- | --- | --- | --- |
| nFix | 15.1000 | 9.4182 | -1.6525 | 65.7674 |
| FFD | 3.1900 | 1.4197 | 0.0079 | 6.9687 |
| GPT | 6.3500 | 5.9089 | -2.9288 | 45.0799 |
| TRT | 5.3100 | 3.6393 | -0.7607 | 23.7844 |
| fixProp | 67.0600 | 26.0551 | 3.4041 | 106.12 |

## Predicted-test feature ranges

| feature | mean | std | min | max |
| --- | --- | --- | --- | --- |
| nFix | 21.4945 | 6.3761 | 0.0000 | 50.9221 |
| FFD | 4.3715 | 0.5644 | 0.0000 | 6.1613 |
| GPT | 8.2804 | 5.4227 | 0.0000 | 40.5412 |
| TRT | 6.9224 | 2.3848 | 0.0000 | 18.5499 |
| GD | 5.1594 | 1.2165 | 0.0000 | 11.4924 |

## Do not mix these scales with ZuCo z-scores

PROVO nFix sits near 15.1 with a percent-like fixProp (mean 67.1). Predicted nFix is also a large unstandardized count. The ZuCo training table is z-scored around 0. Feeding PROVO columns into ``EyeTrackingModel`` without a scaler would let nFix dominate the Linear(5 → 16) layer.

Word-id histograms are just a sanity check that neither file is a single sentence: PROVO unique word_ids=53, predicted unique word_ids=42.
