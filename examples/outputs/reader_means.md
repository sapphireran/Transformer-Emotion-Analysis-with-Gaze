# Reader-level sentence means

Task-1 sentence tables (`ZuCo_et_csv_data/{1..12}_SR.csv`). Subject 3 is
the 299-row packed file. Subject 12's mean pupil is an outlier on the low
side (~298 vs ~800–1300). That survives into `average_data.csv` because
the average is a plain mean across whatever rows share a DataFrame index.

| subject | n_sentences | id_min | id_max | mean_SentLen | mean_omission | mean_nFixations | mean_pupil | mean_TRT | mean_FFD | mean_GPT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 400 | 0 | 399 | 17.8225 | 0.2691 | 1.5462 | 828.5338 | 167.7423 | 106.6034 | 192.1122 |
| 2 | 400 | 0 | 399 | 17.8225 | 0.3728 | 1.6015 | 1354.9090 | 178.9465 | 107.7987 | 243.5122 |
| 3 | 299 | 0 | 298 | 17.7023 | 0.4877 | 1.6006 | 567.9505 | 190.0432 | 113.6645 | 264.0940 |
| 4 | 400 | 0 | 399 | 17.8225 | 0.3291 | 2.0878 | 794.7349 | 248.8023 | 114.4715 | 313.2933 |
| 5 | 400 | 0 | 399 | 17.8225 | 0.2269 | 1.7072 | 967.1812 | 212.6538 | 118.0075 | 239.0781 |
| 6 | 400 | 0 | 399 | 17.8225 | 0.1925 | 1.9294 | 529.6762 | 267.8712 | 137.5570 | 317.7787 |
| 7 | 400 | 0 | 399 | 17.8225 | 0.3165 | 1.3772 | 968.3793 | 160.2901 | 116.8908 | 175.7450 |
| 8 | 400 | 0 | 399 | 17.8225 | 0.3092 | 1.8286 | 540.4838 | 199.7947 | 102.9487 | 245.2292 |
| 9 | 400 | 0 | 399 | 17.8225 | 0.2898 | 1.5448 | 901.1109 | 203.6195 | 131.5618 | 228.7924 |
| 10 | 400 | 0 | 399 | 17.8225 | 0.2577 | 1.8574 | 903.3627 | 238.3155 | 129.2714 | 269.6860 |
| 11 | 400 | 0 | 399 | 17.8225 | 0.3814 | 1.5364 | 827.8504 | 175.2772 | 110.6191 | 199.1548 |
| 12 | 400 | 0 | 399 | 17.8225 | 0.3265 | 1.6064 | 297.6024 | 185.6300 | 111.2050 | 220.1267 |

Pupil min/median/max across readers: 297.6 / 828.2 / 1354.9.

Omission rate is the fraction of words in the sentence with no reported
fixation. Subject 3 is again the high-omission reader, which matches the
word-level skip table.
