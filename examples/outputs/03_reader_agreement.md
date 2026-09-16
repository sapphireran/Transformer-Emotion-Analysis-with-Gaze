# Twelve-reader agreement (ids 0–149)

Only the first 150 sentences are used. After that, reader 3's ids are compacted and pairwise correlations would mix different texts.

| feature | mean CV | mean pairwise r | min r | max r | pairs |
| --- | --- | --- | --- | --- | --- |
| omissionRate | 0.4426 | 0.2840 | -0.0031 | 0.5065 | 66 |
| nFixations | 0.2683 | 0.3619 | 0.1015 | 0.6017 | 66 |
| meanPupilSize | 0.3241 | 0.2274 | -0.2737 | 0.7396 | 66 |
| GD | 0.2106 | 0.3553 | -0.0686 | 0.6493 | 66 |
| TRT | 0.3021 | 0.4371 | 0.2232 | 0.7044 | 66 |
| FFD | 0.1504 | 0.1022 | -0.2020 | 0.4400 | 66 |
| SFD | 0.4260 | 0.1021 | -0.1286 | 0.4111 | 66 |
| GPT | 0.3462 | 0.3318 | 0.0752 | 0.5600 | 66 |

## Notes

SentLen is omitted because it is a property of the text, not the reader: every aligned file has the same length, so correlation is undefined or trivial. Duration measures (GD, TRT, GPT) usually agree more than pupil size. A mean pairwise r around 0.3–0.5 means the readers are not interchangeable, which is why averaging them is a modelling choice rather than a free lunch.
