# Gaze feature glossary

Eye-tracking features look like a pile of acronyms until you walk them
along one word. This page is the glossary I wish I had written when I
named the CSV columns. All **raw** numbers below come from
`ZuCo_et_csv_data/average_data.csv` (sentence level, 12-reader mean) or
`ZuCo_et_csv_data/word/word_averages_v2.csv` (word level, 12-reader mean).

A fixation is a pause of the eyes, typically ~200–300 ms in silent
reading. A saccade is the jump between pauses. Almost every feature here
is "how many pauses" or "how many milliseconds of pauses, under some
rule about regressions."

## Picture of one word

Take the first word of ZuCo sentence 0, `presents`:

| Feature | 12-reader mean | How to read it |
| --- | --- | --- |
| WordLen | 8 | Letters after stripping punctuation |
| nFixations | 2.67 | Readers landed here about two or three times |
| FFD | 135 ms | The first of those landings lasted ~135 ms |
| GD | 161 ms | First pass (before the eyes left the word) lasted ~161 ms |
| TRT | 381 ms | All landings, including later regressions, sum to ~381 ms |
| SFD | 42 ms | Pulled down by trials that were *not* single-fixation |
| GPT | 161 ms | First-pass exit went forward, so GPT ≈ GD here |
| meanPupilSize | 831 | Arbitrary tracker units, not millimeters |

Compare the last content word, `decency`:

| Feature | 12-reader mean |
| --- | --- |
| nFixations | 1.25 |
| FFD | 115 ms |
| GD | (in the same file)  — first pass is modest |
| TRT | 159 ms |
| GPT | **1304 ms** |

GPT exploding on `decency` is the classic wrap-up / spillover pattern:
the eyes entered the word and did not move *past* it for a long time,
usually because this is the end of the sentence or because the reader
was still integrating "dictums of human decency." GPT is not "how long
they stared at this word." It is "how long until they were done with
this region and everything still unresolved behind it."

Function words on the same sentence are almost skipped:

| Word | nFixations | FFD | TRT | GPT |
| --- | --- | --- | --- | --- |
| a (2nd token) | 0.92 | 63 | 89 | 170 |
| to | 0.67 | 50 | 70 | 65 |
| for | 0.42 | 35 | 41 | 41 |
| us | 0.08 | 12 | 12 | 12 |
| the | 0.33 | 24 | 31 | 24 |

`us` at 0.08 fixations means most of the 12 readers never landed on it.
That is omission, not a 12 ms "glance."

Run `python examples/walk_sentence_gaze.py` to reprint this table from
the CSV instead of trusting this page.

---

## Feature by feature

### nFixations / nFix

**Definition.** Count of fixations whose centroid falls on the interest
area (word or, after averaging, the sentence).

**Sentence-level in this repo.** `DataTransformer` sums word-level
nFixations and divides by the number of words that had *any* fixation
reported. So the sentence value is closer to "fixations per fixated
word" than to "fixations per word including skips."

**Raw sentence range (400 rows):** 1.20 – 3.58, mean 1.69.

**When it is large.** Short, dense, or odd sentences. The highest
z-scored nFixations in `combined_sst_et_standard.csv` is sentence 80,
`... a bland murder-on-campus yawner.` at **+6.12**. Three dots of
ellipsis plus a packed insult: readers hit it repeatedly.

**When it is small.** Long sentences that people skim. Sentence 284
(`I have no problem with "difficult" movies...`) sits at **−1.56**.

**Alias.** Full SST files call this `nFix`.

### FFD — first fixation duration

**Definition.** Duration of the *first* fixation on the region, in
milliseconds. If the word was never fixated, the scripts store 0 and
later treat 0 as missing when averaging.

**What it is good for.** Early lexical access: frequency, familiarity,
surprising morphology. It does **not** include re-reading.

**Raw sentence range:** 102 – 166 ms, mean 117, std only 8 ms.
Sentence-level FFD is a very compressed feature. Most of the variance
you see in the z-scored column is a few outliers (sentence 3, "Slow,
silly and unintentionally hilarious.", FFD z = +2.32).

**Word-level** is more informative: `presents` 135 ms vs `a` 32–63 ms.

### GD — gaze duration (first-pass time)

**Definition.** Sum of fixations on the region from first entry until
the eyes first leave it, in any direction. Also called first-pass time.

**Relation to FFD.** GD ≥ FFD always. GD − FFD is extra first-pass
refixations (the reader bounced inside the word before leaving).

**Raw sentence range:** 108 – 273 ms, mean 141.

**Personal reading.** On word-level data, content words with GD ≈ FFD
were processed in one look. Words with GD much larger than FFD were
immediately refixated — often long or unpredictable tokens (`failing`,
`dictums`).

### TRT — total reading time

**Definition.** Sum of *all* fixation durations on the region, including
regressions from the right.

**Relation to GD.** TRT ≥ GD. TRT − GD is time spent coming back later.

**Raw sentence range:** 131 – 427 ms, mean 203. Wider than FFD or GD.
This is the "they kept returning" feature.

**Example.** `presents`: GD 161, TRT 381 → about 220 ms of later
re-reading. `decency`: TRT 159 vs GPT 1304 → they did not pile
fixations on the word itself; they just did not leave the region.

### GPT — go-past time (regression-path duration)

**Definition.** Time from first entering the region until the eyes
first move to the *right* of it. Includes regressions to earlier words.

**Relation to the others.**

- If the reader never regresses, GPT ≈ GD.
- If the reader looks back, GPT can be much larger than TRT on that
  word, because GPT is charging time spent on *earlier* words to this
  one.

**Raw sentence range:** 153 – 587 ms, mean 242. Heaviest right tail of
the five. Sentence 4 (`Ultimately feels empty and unsatisfying, like
swallowing a Communion wafer without the wine.`) has GPT z = **+6.09**,
the maximum in the 400-row file. That is a long simile with a late
sentiment punch; readers go back.

**Personal rule of thumb.** GPT is the feature I look at first when a
review flips polarity in the second clause.

### SFD — single fixation duration

**Definition.** Duration of the only fixation, *if and only if* the
region was fixated exactly once in the trial.

**Why I do not put it in the model.** After averaging 12 readers, a
word that was sometimes skipped, sometimes looked at once, and
sometimes looked at three times becomes a mushy number. Sentence-level
SFD mean is 72 ms with a wide relative spread, but I do not trust it as
a cognitive code. It stays in the CSV.

### omissionRate

**Definition.** Fraction of words in the sentence with no fixation.

**Raw sentence range:** 0.16 – 0.60, mean 0.32. So a typical ZuCo
review in this dump had about one in three words skipped, which is
normal for skilled reading.

**Collinearity.** High omission usually means lower nFixations per
word. That is why the fusion model keeps nFixations and drops this.

### meanPupilSize

**Definition.** Mean pupil size over fixations on the region. Units are
whatever the EyeLink (or equivalent) exported — treat as relative.

**Raw sentence range:** 680 – 970, mean 797, std 61.

**Use.** Arousal / load / lighting. I plot it. I do not train on it in
the current scripts.

### SentLen / WordLen

Sentence length in words, word length in characters after
`re.sub('[^\w\s]', '', word.content)` in `DataTransformer`. Leading
words are lowercased; later tokens keep internal caps (`Beckett` in
the predicted-gaze file). Length is a confounder for every reading-time
measure. Any analysis that says "negative sentences have higher TRT"
has to survive a length check. Mean ZuCo sentence is **17.8 words**
(min 3, max 43). Mean full-SST sentence is **19.2 words**.

---

## Scaling, in one worked column

Sentence 0, raw nFixations = 2.1585.

Across 400 sentences, nFixations has mean 1.6870 and std 0.3101.

```
z = (2.1585 - 1.6870) / 0.3101 ≈ 1.521
```

The value in `combined_sst_et_standard.csv` for sentence 0 is
`1.520856…`. That is the number `EyeTrackingModel` sees — not 2.16
fixations, not 2.16 milliseconds.

Min-max for the same cell:

```
(2.1585 - 1.2046) / (3.5833 - 1.2046) ≈ 0.401
```

which matches `combined_sst_et_min_max.csv`. I prefer z-scores for the
linear gaze tower because a 6σ GPT (sentence 4) stays visibly large
instead of saturating at 1.

## Predicted gaze is in a different unit system

`gaze_prediction/data/prediction_test.csv` stores nFix around mean
**21.5**, FFD around **4.4**, GPT around **8.3**. Those are not
milliseconds. `gaze_prediction/data/convert_zuco_data.py` rescales
nFixations to 0–100 from its own min/max, and rescales FFD/GPT/TRT/GD
onto a *shared* 0–100 range. PROVO (`provo.csv`) is in yet another
scaled space (nFix mean 15.1, `fixProp` mean 67).

Do not concatenate a ZuCo z-score and a predicted-gaze row. The fusion
model never does that: Track A and Track B each stay inside one file.

## Correlation warning

On both the raw ZuCo averages and the predicted-gaze plots in
`result/`, nFix / GD / TRT / GPT move together. FFD is a little more
independent (first look only) but still positively correlated. If you
need a single diagnostic number for "reading effort," TRT or GPT is
enough. The five-feature vector is for compatibility with the papers
that report all of them, and for the slim chance the linear layer
finds a contrast (for example high GPT, low TRT = regression-heavy,
not dwell-heavy).
