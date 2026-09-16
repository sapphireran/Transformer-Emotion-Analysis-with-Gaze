# Worked example: sentence 0

The first ZuCo ∩ SST row is a useful walkthrough because every derived table
contains it.

## Text and label

From `ZuCo_SST_data/ssts_ZuCo.csv`:

```
Presents a good case while failing to provide a reason for us to care
beyond the very basic dictums of human decency.
```

`sentiment_label = 1` (neutral). That matches the hedge: praise (`good
case`) and dismissal (`failing`, `unsatisfying` territory) in one sentence.

## Sentence-level human gaze

Raw subject-averaged values (`average_data.csv`, id 0):

| SentLen | omissionRate | nFixations | meanPupilSize | GD | TRT | FFD | SFD | GPT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 22 | 0.254 | 2.16 | 851 | 143 | 262 | 116 | 48 | 332 |

Compared with the 400-sentence raw means (18 words, omission 0.32,
nFix 1.69, TRT 203, GPT 242) this sentence is **longer, less skipped, and
heavier on rereading**. Standard-scaled (`combined_sst_et_standard.csv`)
that shows up as `nFixations = 1.52`, `TRT = 1.25`, `GPT = 1.59`,
`SFD = -2.16` — many fixations, almost no single-fixation words.

Min-max places the same row at `nFixations = 0.40`, `GPT = 0.41`, still
above the min-max means (~0.20 for those channels).

## Word-level profile

22 tokens in `word_averages_v2.csv` with `Sent_ID = 0_NR`.
`examples/05_word_level_profiles.py --sentence-id 0_NR` reprints this table
and writes `examples/output/word_profile_0.png`.

| word | nFix | FFD | GD | TRT | GPT |
| --- | ---: | ---: | ---: | ---: | ---: |
| presents | 2.67 | 135 | 161 | 381 | 161 |
| a | 0.92 | 63 | 63 | 89 | 170 |
| good | 2.58 | 97 | 115 | 302 | 219 |
| case | 1.75 | 105 | 150 | 241 | 240 |
| while | 2.50 | 121 | 158 | 296 | 303 |
| **failing** | **3.08** | 104 | 167 | **387** | 201 |
| to | 0.67 | 50 | 50 | 70 | 65 |
| provide | 2.83 | 145 | 162 | 345 | 227 |
| a | 0.50 | 32 | 32 | 54 | 32 |
| **reason** | 2.92 | 125 | 197 | 340 | 354 |
| for | 0.42 | 35 | 41 | 41 | 41 |
| us | 0.08 | 12 | 12 | 12 | 12 |
| to | 0.75 | 71 | 79 | 101 | 163 |
| care | 1.83 | 107 | 117 | 221 | 310 |
| **beyond** | 2.17 | 93 | 112 | 236 | **507** |
| the | 0.33 | 24 | 24 | 31 | 24 |
| very | 1.67 | 99 | 121 | 195 | 321 |
| basic | 2.08 | 109 | 130 | 256 | 206 |
| dictums | 2.42 | 112 | 181 | 303 | 208 |
| of | 0.42 | 50 | 50 | 50 | 50 |
| human | 1.58 | 119 | 130 | 212 | 295 |
| **decency** | 1.25 | 115 | 121 | 159 | **1304** |

Function words (`a`, `to`, `for`, `us`, `the`, `of`) are barely fixated.
Content that carries the verdict — `failing`, `reason`, `provide`,
`presents` — collects the total reading time. `decency` is not the longest
TRT, but it has a **go-past time of 1.3 seconds**: readers entered the last
word and spent a long time going back before leaving the sentence. That is
a regression-path event the sentence mean (`GPT = 332 ms`) only hints at.

## What the fused model actually receives

`model_ZuCo_SST.py` does not see this 22 × 5 word matrix. It sees one
standard-scaled vector:

```
nFixations=1.521  FFD=-0.078  GPT=1.587  TRT=1.248  GD=0.076
```

plus the RoBERTa pooler for the full string. The linger on `decency` and
the extra fixations on `failing` are compressed into those five numbers.

A token-level fusion model would instead add a 5-d gaze vector (or a
predicted one) to each subword. That is out of scope for the current
scripts; the word table is here so you can see what is being thrown away.

## Reconstructing the string

`sentence_profile` joins the `Word` column:

```
presents a good case while failing to provide a reason for us to care
beyond the very basic dictums of human decency
```

Punctuation from the SST original is gone — `DataTransformer` strips it
with `re.sub('[^\w\s]', '', word.content)` and lowercases only the first
token. Aligning these tokens to RoBERTa BPE would need a second alignment
step that the training code never performs.
