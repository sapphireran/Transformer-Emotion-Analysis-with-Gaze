# Worked examples from the CSVs

Every number on this page was read from the checked-in files, not from
memory. The same walks are automated in `examples/walk_sentence_gaze.py`.

Labels again: `0` negative, `1` neutral, `2` positive.

---

## Example 1 — Sentence 0, the pipeline poster child

**Text.** *Presents a good case while failing to provide a reason for
us to care beyond the very basic dictums of human decency.*

**Label.** `1` (neutral). That is the right SST call: praise ("good
case") then withdrawal ("failing to provide a reason").

**Sentence-level raw gaze** (`average_data.csv`, 12-reader mean):

| SentLen | omissionRate | nFixations | pupil | GD | TRT | FFD | SFD | GPT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 22 | 0.254 | 2.16 | 851 | 143 | 262 | 116 | 48 | 332 |

Compared to the 400-sentence means (1.69 fix, 203 ms TRT, 242 ms GPT)
this review is **re-read more than average**. After z-scoring
(`combined_sst_et_standard.csv`):

| nFixations | FFD | GPT | TRT | GD |
| --- | --- | --- | --- | --- |
| +1.52 | −0.08 | +1.59 | +1.25 | +0.08 |

The fusion model (Track A) therefore sees: first-pass duration is
ordinary, but people came back (TRT) and took a long time to get
*past* the sentence (GPT). That is the "mixed review, readers
integrate a contrast" pattern.

### Word-level story (`word_averages_v2.csv`, `Sent_ID = 0_NR`)

```
id  word        nFix   FFD    TRT    GPT    len
 0  presents    2.67  135   381   161     8
 1  a           0.92   63    89   170     1
 2  good        2.58   97   302   219     4
 3  case        1.75  105   241   240     4
 4  while       2.50  121   296   303     5
 5  failing     3.08  104   387   201     7
 6  to          0.67   50    70    65     2
 7  provide     2.83  145   345   227     7
 8  a           0.50   32    54    32     1
 9  reason      2.92  125   340   354     6
10  for         0.42   35    41    41     3
11  us          0.08   12    12    12     2
12  to          0.75   71   101   163     2
13  care        1.83  107   221   310     4
14  beyond      2.17   93   236   507     6
15  the         0.33   24    31    24     3
16  very        1.67   99   195   321     4
17  basic       2.08  109   256   206     5
18  dictums     2.42  112   303   208     7
19  of          0.42   50    50    50     2
20  human       1.58  119   212   295     5
21  decency     1.25  115   159  1304     7
```

What I take from this table when I am not rushing:

1. **Sentiment-bearing content words are fixated ~2–3 times**
   (`presents`, `good`, `failing`, `provide`, `reason`, `dictums`).
2. **`failing` is the hottest nFix (3.08) and TRT (387 ms).** That is
   the polarity flip. A token-aligned model would have a chance to
   put weight here. Late fusion only sees that *somewhere* in the
   sentence people lingered.
3. **`good` is also heavy (TRT 302).** The contrast is visible in gaze
   as two peaks, not one.
4. **`us` is omitted (nFix 0.08).** Most readers never landed on it.
   `DataTransformer` would count this word as not-fixated when
   building `nwords_fixated`.
5. **`decency` GPT = 1304 ms** with only 1.25 fixations and TRT 159.
   Sentence wrap-up: the clock keeps running until the eyes leave the
   last region, which is "never, this is the end." Do not interpret
   1304 ms as "decency is a 1.3 second word."
6. **`beyond` GPT = 507 ms** is a more honest mid-sentence go-past
   spike — readers look back before committing to the final clause.

If I were writing a figure caption: *Word-level total reading time
peaks on the evaluative nouns and on the negation (`failing`);
go-past time then balloons at the sentence-final wrap-up.*

---

## Example 2 — Sentence 4, maximum go-past

**Text.** *Ultimately feels empty and unsatisfying, like swallowing a
Communion wafer without the wine.*

**Label.** `0` (negative).

**Z-scored gaze:**

| omissionRate | nFixations | pupil | GD | TRT | FFD | SFD | GPT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| −1.33 | **+4.05** | +0.48 | +1.82 | +3.21 | −0.15 | −2.87 | **+6.09** |

This is the **maximum GPT** in the 400-row standardized file and the
second-highest nFixations (after sentence 80). Raw GPT on the average
table is **587 ms** (the raw max).

Word-level table (`python3 examples/walk_sentence_gaze.py --sentence-id 4`)
is doing more work than the sentence text suggests:

| Word (as stored) | nFix | TRT | GPT | notes |
| --- | --- | --- | --- | --- |
| ultimately | 3.75 | 488 | 198 | long opener |
| feels | 3.25 | 388 | 264 | |
| **emp11111ty** | **7.25** | **920** | 617 | *not* the string `empty` — see below |
| unsatisfying | 3.83 | 503 | 593 | real adjective, heavy |
| Communion | 3.00 | 312 | **1154** | mid-sentence go-past peak |
| without | 1.92 | 194 | 998 | second GPT spike |
| wine | 1.08 | 120 | 707 | wrap-up, GPT > TRT |
| a | 0.00 | 0 | 0 | omitted by every reader in the mean |

The token `emp11111ty` (WordLen 10) is already in every
`ZuCo_et_csv_data/word/{1-12}_SR.csv` for `Sent_ID=4_NR`. The
sentence-level text still says `empty`. So the 12-reader mean is
averaging a **mangled interest-area label**, not the word `empty`.
Subject 12 even has 13 fixations on that area. I am leaving the CSV
alone in this documentation pass; I will not use "readers stared at
*empty*" as a finding until I know whether the tracker IA was
`empty` and the export inserted `11111`, or the IA really was that
string on screen.

The *usable* word-level story is still the simile: GPT peaks on
`Communion` (1154 ms) and `without` (998 ms) while `a` is skipped.
A text-only model already has `empty` / `unsatisfying`. Gaze is
piling on the same negative direction plus a long integration of the
wafer/wine clause. Fusion helping here would not surprise me. Fusion
helping on a *positive* sentence with the same GPT would be more
interesting.

---

## Example 3 — Three-word reviews

ZuCo is not only long clauses.

| id | label | text | nFix z | GPT z |
| --- | --- | --- | --- | --- |
| 40 | 2 | Reassuring, retro uplifter. | +1.94 | +3.14 |
| 135 | 0 | under-rehearsed and lifeless | +2.94 | +4.09 |
| 339 | 2 | An exhilarating experience. | +0.81 | +1.62 |

Short sentences have fewer places to skip, so nFixations per fixated
word goes up (the denominator in `DataTransformer` is small). Sentence
135 is a two-adjective slam with **no verb**; readers still produce a
very high GPT z. Sentence 339 is almost a template positive and sits
closer to the mean.

For a transformer, three WordPieces-and-change is easy. For gaze,
three-word sentences are **high-variance**: one skipped word changes
omissionRate a lot. I would not pick a winner between fusion and
text-only on this subset without looking at residuals.

---

## Example 4 — High nFix, low lexical drama

**Sentence 80.** `... a bland murder-on-campus yawner.`

**Label.** `0`. **nFixations z = +6.12** (the maximum). Raw SentLen = 5.

Word-level (`walk_sentence_gaze.py --sentence-id 80`):

| Word (as stored) | nFix | TRT | GPT |
| --- | --- | --- | --- |
| unknown | 0.00 | 0 | 0 |
| a | 0.08 | 9 | 9 |
| bland | 1.17 | 167 | 109 |
| **murderoncampus** | **6.67** | **796** | **588** |
| yawner | 1.42 | 158 | 466 |

Two pipeline fingerprints in one row:

1. The leading `...` was stripped to an empty token and stored as
   `unknown` (see `word/get_average.py`).
2. Hyphens were stripped (`re.sub('[^\w\s]', '', ...)`), so the
   compound became the 14-letter blob `murderoncampus`. That blob
   soaks almost all of the sentence's fixations.

Extreme nFix is therefore *not* "readers hated `yawner`." It is a
five-word sentence whose one content compound was glued together
and then used as the denominator-heavy IA. I would not use sentence
80 as the slide that says "gaze detects negativity." I would use it
as the slide that says "always look at the word table before you
quote a sentence-level z-score."

---

## Example 5 — Low nFix, long sentence

**Sentence 284.** Starts *I have no problem with "difficult" movies,
or movies that ask the audience to meet them halfway…*

**Label.** `1`. **nFixations z = −1.56** (the minimum).

Long, chatty, meta. Readers can skim. The transformer sees a lot of
tokens; gaze says "this was easy / skippable." If fusion ever *hurts*
accuracy, this is the shape of sentence I would inspect first: text
is doing the work, gaze is a near-zero (or negative) "effort" cue
that the linear layer might over-interpret.

---

## Example 6 — Full SST row, predicted gaze

From `SST_data/train_full_sst.csv`, first row in file order (not
sentence_id order):

```
sentence_id   11562
sentence      Plays like a volatile and overlong W magazine fashion spread .
label         0
nFix          +0.767
GD            +0.945
TRT           +0.670
FFD           +0.740
GPT           +0.729
```

All five predicted features are positive and similar. That is the
correlation warning from the glossary: the predictor is outputting
a single "somewhat high effort" direction, copied five times with
jitter. Contrast with a low-effort row in the same file
(`sentence_id` 4457, the long "actresses may have worked up a back
story…" sentence): nFix **−1.03**, TRT **−1.15**. The predicted
vector is again internally consistent.

I do **not** read these as milliseconds. I read them as "the gaze
head on Track B is being fed a 5-D copy of a scalar." That still
might help if the scalar correlates with label after the text
representation is accounted for. It might also be a length / rarity
proxy the transformer already has.

---

## Example 7 — PROVO vs predicted scatter

`result/provo_data_scatter_hist_plots.png` (and the train/test twins)
show nFix, FFD, GPT, TRT, GD (or `fixProp` on PROVO) as a lower
triangle of scatter plots plus histograms.

What I use those figures for:

- **Sanity.** No mass of negative milliseconds. A few scaled PROVO
  values go slightly negative (`nFix` min −1.65, `GPT` min −2.93)
  because that file is in a standardized / shifted space, not raw ms.
- **Collinearity.** TRT vs nFix and GD vs TRT are almost linear.
  FFD vs nFix is curved (first look saturates; extra fixations show
  up in the other features).
- **Train vs test plots** (`train_data_scatter_hist_plots.png`,
  `test_data_scatter_hist_plots.png`) should look like the same
  cloud. If a later predictor dump ever breaks that, the figure is
  how I notice.

I am not treating the PNG titles ("ProvoDataScatterandHistogramPlototles")
as publication figures. They are lab scratch.

---

## How to reprint these

```bash
python3 examples/walk_sentence_gaze.py                # sentence 0
python3 examples/walk_sentence_gaze.py --sentence-id 4
python3 examples/walk_sentence_gaze.py --sentence-id 80
python3 examples/walk_sentence_gaze.py --sentence-id 135
python3 examples/inspect_datasets.py                  # counts + ranges
```

If a CSV is regenerated and these numbers move, believe the script
output and update this page.
