# Word → sentence walkthrough (id 0)

**Text:** Presents a good case while failing to provide a reason for us to care beyond the very basic dictums of human decency.

**Label:** 1

Word-average tokens: 22 (zeros in nFixations: 0). Mean word nFixations=1.610, TRT=196.515.

## First twelve word-average rows

| word_id | word | nFixations | FFD | TRT | WordLen |
| --- | --- | --- | --- | --- | --- |
| 0 | presents | 2.6667 | 134.58 | 381.17 | 8.0000 |
| 1 | a | 0.9167 | 62.9167 | 88.9167 | 1.0000 |
| 2 | good | 2.5833 | 96.9167 | 301.92 | 4.0000 |
| 3 | case | 1.7500 | 105.33 | 241.00 | 4.0000 |
| 4 | while | 2.5000 | 121.42 | 296.25 | 5.0000 |
| 5 | failing | 3.0833 | 104.25 | 386.83 | 7.0000 |
| 6 | to | 0.6667 | 50.1667 | 69.5833 | 2.0000 |
| 7 | provide | 2.8333 | 144.50 | 345.08 | 7.0000 |
| 8 | a | 0.5000 | 31.5000 | 54.3333 | 1.0000 |
| 9 | reason | 2.9167 | 124.75 | 340.33 | 6.0000 |
| 10 | for | 0.4167 | 35.0000 | 41.2500 | 3.0000 |
| 11 | us | 0.0833 | 11.8333 | 11.8333 | 2.0000 |

## Sentence-level published numbers

| table | nFixations | TRT | FFD | GPT |
| --- | --- | --- | --- | --- |
| average_data.csv (raw-ish mean) | 2.1585 | 262.41 | 116.25 | 331.68 |
| standard_scaled_average_data.csv | 1.5209 | 1.2480 | -0.0784 | 1.5875 |
| combined_sst_et_standard.csv (train) | 1.5209 | 1.2480 | -0.0784 | 1.5875 |

## Reader 1 vs remapped reader 3, same sentence

Reader 1 words on 0_NR: 22. Reader 3 words whose remapped Sent_ID is 0_NR: 22. First tokens: reader1=presents, reader3=presents.

## Do not expect word-means to equal sentence-means

``DataTransformer`` builds sentence features by summing word measures and dividing by the number of *fixated* words, not by ``SentLen``. Omitted words (nFixations=0) therefore drop out of the sentence average. The word table keeps those zeros. That is why ``mean word nFixations`` above is not a copy of ``average_data``.
