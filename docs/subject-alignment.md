# Subject alignment

This is the main personal finding in this pass. It is not a cleanup of
the training scripts — it is a reading of `3_SR.csv` against the other
eleven readers.

## What the producer does

`utils_ZuCo.DataTransformer` walks the MATLAB `sentenceData` array.
For task 1, **subject index 2** (one-based file `3_SR.csv`) it skips
original sentences `150..249` and `399`, then writes whatever is left
with a compacted `id` of `0..298`.

```
original  0……149  150……249  250……398  399
file 3    0……149   (dropped) 150……298  (dropped)
```

`compact_to_original_subject3(150) == 250`. Tests lock this mapping
down in `tests/test_subjects.py`.

The same compaction shows up at word level. In `word/3_SR.csv` the
sentence tagged `150_NR` starts with `the`, which is the first token
of original sentence 250 in `word/1_SR.csv`, not sentence 150 (`its`).

## What the averager does

`get_average_sentence_level.py` concatenates `{1..12}_SR.csv` and
groups by row index. From id 150 onward, reader 3 is contributing a
**different sentence**.

`examples/02_subject_alignment.py` compares that index mean to a
remapped mean:

| region | ids | what happens | mean \|Δ nFixations\| |
| --- | --- | --- | ---: |
| aligned | 0–149 | all 12 readers, same text | 0 |
| wrong sentence | 150–249 | reader 3 donates 250–349 | 0.030 |
| shifted both ways | 250–298 | reader 3 donates 350–398; also still present under the compact id | 0.038 |
| reader 3 missing from index | 299–398 | 11-reader index mean vs 12-reader remapped mean | 0.028 |
| skipped | 399 | reader 3 absent both ways | 0 |

246 of 400 sentences have a nonzero nFixations delta. The largest
\|Δ\| is about 0.21. One reader out of twelve cannot move the mean
very far, so a model trained on `combined_sst_et_standard.csv` is not
ruined — but the average is not the average it claims to be.

`SentLen` makes the bug obvious without any gaze theory. At id 150
the index-mean length is 9.5 (eleven readers at 9, reader 3 at 15).
The remapped mean is 9.0.

## Clean window for agreement

Pairwise reader correlations are only honest on ids **0–149**.
`examples/03_reader_agreement.py` on that window:

| feature | mean CV | mean pairwise r |
| --- | ---: | ---: |
| TRT | 0.30 | 0.44 |
| nFixations | 0.27 | 0.36 |
| GD | 0.21 | 0.36 |
| GPT | 0.35 | 0.33 |
| omissionRate | 0.44 | 0.28 |
| meanPupilSize | 0.32 | 0.23 |
| FFD | 0.15 | 0.10 |
| SFD | 0.43 | 0.10 |

Readers are not interchangeable. Averaging them is a modelling choice.
FFD agrees surprisingly little across people; TRT agrees the most.

## Word-level averages have the same index bug

`ZuCo_et_csv_data/word/get_average.py` concatenates the twelve word
files and groups by row index, then pastes reader 1's `Sent_ID` /
`Word` columns on the left. After the skip region, reader 3's words
are averaged into the wrong tokens.

`zuco_lab.subjects.remap_word_sent_id` rewrites `150_NR` → `250_NR`
for subject 3 if you want a corrected join. The examples do not
rewrite `word_averages_v2.csv`.

## Why this was left unpatched

A silent re-average would make a later `model_ZuCo_SST.py` run
irreconcilable with any number already written down from this clone.
The personal layer reports the delta and keeps the published tables.
