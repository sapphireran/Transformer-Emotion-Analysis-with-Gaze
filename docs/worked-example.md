# Worked example: sentence 3

One ZuCo ∩ SST row, from the string a subject saw to the five floats
`EyeTrackingModel` concatenates with the RoBERTa pooler.

Re-print this page from the CSVs:

```bash
python3 examples/sentence_gaze_walkthrough.py --sentence-id 3
```

---

## The text and the label

From `ZuCo_SST_data/ssts_ZuCo.csv`:

| Field | Value |
| --- | --- |
| `sentence_id` | 3 |
| `sentence` | `Slow, silly and unintentionally hilarious.` |
| `sentiment_label` | `1` (neutral) |

Five tokens after ZuCo’s cleaner (punctuation gone, first word
lowercased):

```text
slow | silly | and | unintentionally | hilarious
```

`SentLen = 5` in `average_data.csv`. A human reader can finish this in
a second; the interesting part is *where* they hesitated.

Neutral is the right SST tag: the line is a stacked adjective joke, not
a clean thumbs-up or thumbs-down.

---

## Word-level subject mean

`ZuCo_et_csv_data/word/word_averages_v2.csv`, `Sent_ID = 3_NR`
(milliseconds / counts, mean of 12 subjects, zeros kept):

| Word | nFix | GD | TRT | FFD | SFD | GPT | chars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slow | 1.92 | 138 | 247 | 131 | 44 | 138 | 4 |
| silly | 2.67 | 161 | 334 | 113 | 14 | 242 | 5 |
| and | 1.67 | 87 | 174 | 87 | 39 | 216 | 3 |
| unintentionally | 5.42 | 292 | **802** | 173 | 0 | 441 | 15 |
| hilarious | 2.33 | 182 | 314 | 138 | 23 | **1033** | 9 |

Read this as a cartoon of the sentence:

1. **slow** — ordinary first pass. GPT = GD, so readers did not go
   backwards before leaving it.
2. **silly** — a bit more re-reading (`TRT > GD`, GPT up).
3. **and** — short function word, small GD, but GPT > GD: some eyes
   bounced left after landing here.
4. **unintentionally** — the load-bearing adverb. Longest word, most
   fixations, huge total time, **SFD = 0** (nobody got it in a single
   fixation on average). First-pass GD is already 292 ms; later visits
   add another ~500 ms of TRT.
5. **hilarious** — first-pass GD looks normal, but GPT is over a
   second. Readers entered the last word and then **regressed** (often
   back through *unintentionally*) before the eyes left the sentence.

That GPT spike on the last word is wrap-up plus a punchline: the
adjective that retroactively colours *slow* and *silly*.

---

## Two actual subjects

Same five rows in `word/1_SR.csv` and `word/2_SR.csv`.

**Subject 1** fixated everything:

| Word | nFix | GD | TRT | GPT |
| --- | ---: | ---: | ---: | ---: |
| slow | 3 | 162 | 329 | 162 |
| silly | 2 | 97 | 150 | 317 |
| and | 1 | 103 | 103 | 103 |
| unintentionally | 3 | 121 | 288 | 121 |
| hilarious | 2 | 139 | 306 | 473 |

**Subject 2** *skipped* `slow` (all zeros) and spent six fixations plus
707 ms TRT on `unintentionally`. `hilarious` was a single 91 ms
fixation, but GPT is 900 ms — almost all of that time is regressions
*before* the trial moved past the last word.

Averaging these two already shows why sentence-level `nFixations` is
high and why `omissionRate` is not zero: one reader omitted the first
word, and both spent extra time on the 15-letter adverb.

---

## Sentence-level raw mean

`ZuCo_et_csv_data/average_data.csv`, `id = 3`:

| Column | Value | Units |
| --- | ---: | --- |
| SentLen | 5.00 | words |
| omissionRate | 0.267 | fraction |
| nFixations | 2.97 | per fixated word |
| meanPupilSize | 877 | EyeLink pupil units |
| GD | 182 | ms |
| TRT | 396 | ms |
| FFD | 135 | ms |
| SFD | 39 | ms |
| GPT | 440 | ms |

Compared with the 400-sentence cloud this row is a **long, sticky
read** of a short string: TRT and GPT sit well above the typical
~200–300 ms sentence means (see `examples/gaze_feature_stats.py`).

`DataTransformer` built these by summing word features and dividing by
the number of *fixated* words, then the averager took the mean across
subjects. You cannot recover the 802 ms on *unintentionally* from 396
ms TRT alone. The classifier never sees the word table.

---

## What the fusion head actually gets

`model_ZuCo_SST.py` reads
`combined_sst_et_standard.csv` and keeps five z-scores:

| Column | z-score on sentence 3 |
| --- | ---: |
| nFixations | **+4.13** |
| FFD | +2.32 |
| GPT | +3.50 |
| TRT | **+4.04** |
| GD | +1.89 |

Dropped: `omissionRate -0.79`, `meanPupilSize +1.31`, `SFD -2.98`.

SFD is *low* because the interesting words were re-fixated (single-
fixation duration does not apply). A model that only saw SFD would
think this sentence was easy. The five-feature subset keeps the
“readers stuck here” story.

The gaze MLP does:

\[
g_{16} = W_g\,[4.13,\; 2.32,\; 3.50,\; 4.04,\; 1.89]^\top + b_g
\]

and concatenates \(g_{16}\) with a 768-d encoding of the string
`Slow, silly and unintentionally hilarious.`

RoBERTa still has to decide **neutral vs positive**: the lexical cue
*hilarious* pulls positive; *slow* / *silly* pull negative; gaze only
says “this line was effortful”. Effort is not polarity. If fusion
helps on this row, it is as a tie-breaker or a regulariser, not as a
label.

Min-max twin (`combined_sst_et_min_max.csv`), same sentence:

| Column | min-max |
| --- | ---: |
| nFixations | 0.741 |
| FFD | 0.527 |
| GPT | 0.661 |
| TRT | 0.895 |
| GD | 0.450 |
| SFD | 0.000 |

SFD hits the floor of the 400-row min-max. Do not mix this row with the
z-score row inside one batch.

---

## Full-SST counterpart (different claim)

There is no measured gaze for this string in `SST_data/`. If the same
review appears in the 11k table, its `nFix, GD, TRT, FFD, GPT` are
**projected**. Do not paste the +4.13 nFixations into a sentence about
“readers”. The walkthrough script will say `measured` vs `projected`
explicitly.

---

## What to look at next

- `--sentence-id 0` — longer, mixed review; calmer z-scores.
- `--sentence-id 4` — `GPT` z ≈ +6.1 on a communion-wafer simile.
- `examples/word_level_skip_analysis.py` — how often `slow`-style
  skips happen corpus-wide.
- `examples/toy_text_gaze_fusion.py` — whether five z-scores like
  these move a hashed-text logistic model at all.
