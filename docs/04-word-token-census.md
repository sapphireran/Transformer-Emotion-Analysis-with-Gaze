# Word-token census

`DataTransformer` strips punctuation with `re.sub('[^\w\s]', '', …)` and
lowercases only the first token of a sentence. Hyphens, slashes, and
apostrophe-adjacent marks disappear. The word-average file keeps the
mangled form and a `WordLen` that is `len(token)` **after** the strip —
except when the token was empty and later filled with `unknown`, which
stores `WordLen = 0`.

## Hand-checked glued forms

| Sent_ID | token in `word_averages_v2.csv` | intended reading | notes |
| --- | --- | --- | --- |
| 4_NR | `emp11111ty` | empty | already mangled in the per-reader files; WordLen 10 |
| 22_NR | `20thcentury` | 20th-century | hyphen dropped |
| 69_NR | `massmurdering` | mass-murdering | hyphen dropped |
| 80_NR | `murderoncampus` | murder-on-campus | six-sigma-ish nFix on a glued content word |
| 199_NR | `allwiseguysallthetime` | all-wise-guys-all-the-time | WordLen 21 |
| 358_NR | `under10` | under-10 | hyphen dropped |

Sentence 4 in `ssts_ZuCo.csv` still says *empty*. Every `word/{k}_SR.csv`
already stores `emp11111ty`. Whatever produced the MATLAB `word.content`
did it before this repo's regex.

## Counts on this clone

- 28 tokens contain a digit (`10`, `1978`, `3D`, `emp11111ty`, …).
- 55 rows have `WordLen != len(Word)`. They are the `unknown`
  placeholders: the original empty token had length 0, then
  `get_average.py` filled the word column with `'unknown'` and left
  `WordLen` at 0.
- `unknown` appears on 55 sentence ids (some sentences have more than
  one). Those slots are punctuation-only or stripped-empty tokens, not
  mystery vocabulary.

## Why it matters for predicted gaze

`SST_data/convert_sst_to_et.py` tokenizes with NLTK and then keeps only
`^[A-Za-z]+$`. Hyphenated SST tokens vanish or split differently from
ZuCo's regex. A word-level gaze predictor trained on `word_averages_v2`
and applied to `sst_et_test.csv` is therefore aligning two incompatible
tokenizations. Sentence-level means hide that; they do not fix it.

Walk the three short examples with `python3 examples/09_sentence_walkthrough.py`.
