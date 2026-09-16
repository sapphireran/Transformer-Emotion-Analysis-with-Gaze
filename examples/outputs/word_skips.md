# Word-level skips

A skip here is `nFixations == 0` on the committed word-level `*_SR.csv`
files. Subject 3 is the short table (299 sentences, packed ids) *and* the
highest skip rate. That is a second reason not to row-average it with the
other eleven readers.

| subject | n_tokens | n_sentences | skip_rate | mean_nFix | mean_TRT | mean_WordLen | empty_tokens | r_nFix_WordLen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 7129 | 400 | 0.2832 | 1.0518 | 112.5692 | 4.8550 | 55 | 0.4988 |
| 2 | 7129 | 400 | 0.3886 | 0.9515 | 103.8007 | 4.8550 | 55 | 0.4415 |
| 3 | 5293 | 299 | 0.5250 | 0.7312 | 84.2044 | 4.8371 | 43 | 0.3846 |
| 4 | 7129 | 400 | 0.3351 | 1.3532 | 156.4214 | 4.8550 | 55 | 0.4817 |
| 5 | 7129 | 400 | 0.2319 | 1.2702 | 153.7409 | 4.8550 | 55 | 0.5029 |
| 6 | 7129 | 400 | 0.2062 | 1.4929 | 203.4047 | 4.8550 | 55 | 0.4664 |
| 7 | 7129 | 400 | 0.3271 | 0.8958 | 103.1264 | 4.8550 | 55 | 0.4820 |
| 8 | 7129 | 400 | 0.3226 | 1.2079 | 129.9184 | 4.8550 | 55 | 0.4783 |
| 9 | 7129 | 400 | 0.2961 | 1.0199 | 132.0551 | 4.8550 | 55 | 0.5381 |
| 10 | 7129 | 400 | 0.2647 | 1.3442 | 172.1112 | 4.8550 | 55 | 0.5586 |
| 11 | 7129 | 400 | 0.3928 | 0.9105 | 100.4675 | 4.8550 | 55 | 0.5189 |
| 12 | 7129 | 400 | 0.3382 | 1.0324 | 114.3113 | 4.8550 | 55 | 0.4843 |

Mean skip rate excluding subject 3: 0.3079.
Subject 3 skip rate: 0.5250.

Empty `Word` cells come from `re.sub('[^\w\s]', '', word.content)` in
`utils_ZuCo.py` wiping tokens that were only punctuation. `WordLen` is 0
on those rows. They are harmless for sentence-level means but they do
show up as `'unknown'` once `word/get_average.py` fills nulls.
