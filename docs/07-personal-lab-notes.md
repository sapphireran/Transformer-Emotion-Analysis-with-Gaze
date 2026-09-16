# Personal lab notes, quirks, and next questions

A running list. Not a ticket tracker. Items marked **trap** have
already wasted time.

---

## Traps in the original scripts

### 1. Test metrics on full SST are last-batch only (**trap**)

`model_full_SST.py` test loop assigns `all_preds = ...` instead of
`.extend(...)`. Validation is correct. See
`05-training-and-evaluation.md`. Do not quote the test printout.

### 2. `et_csv_data` vs `ZuCo_et_csv_data` (**trap**)

`read_ZuCo_mat.py` and `get_average_sentence_level.py` write/read
`et_csv_data`. Git has `ZuCo_et_csv_data/`. Re-running those scripts
in a clean clone will fail or write a second folder.

### 3. Windows path separators in `get_matfiles`

```python
subdir='\\ZuCo_mat_data\\'
path = os.getcwd() + subdir + task
```

Works accidentally on some Python/Windows setups; wrong on Linux.
Use `os.path.join`.

### 4. Subject 2 is a different length

Task-1 subject 2 drops ~101 sentences. Averaging "by row index"
across 12 files is only valid if every file has 400 rows. The
checked-in `average_data.csv` has 400 rows. Whatever produced it
is not guaranteed to be a naive `groupby(level=0).mean()` on the
checked-in `*_SR.csv` files if those individual files differ in
length. I have not re-derived the average in this environment.
If I ever need to regenerate it, I will average **by sentence
id**, inner-joining subjects, not by row number.

### 5. Accuracy commented as F1

`best_val_acc` is accuracy. The Chinese comment and the save
printout say F1. Cosmetic, but I have already misread it once.

### 6. `spilt.py` in two folders

Typo for `split`. Both copies are the same 80/10/10 logic on
different CSVs. The ZuCo one is unused by the training script.

### 7. `CrossEntropyLoss()` constructed per batch

Harmless. Move it next to the optimizer if I touch that loop.

### 8. No `models/` directory

`torch.save(..., 'models/best_...pth')` assumes the folder exists.

### 9. Unused imports

`utils_ZuCo.py` imports `gzip`, `math`, `scipy` (in addition to
`scipy.io`). `model_full_SST.py` imports `Dataset` from
`datasets` only to tokenize. Not worth a cleanup commit by
itself.

### 10. `convert_full_SST.py` vs `save_SST_data.py`

Two scripts, one idea. The former writes `ssts_ZuCo.csv` with an
integer `sentence_id`; the latter writes `output.csv` with the
filename stem as a string. I use `ssts_ZuCo.csv`.

---

## Things that look like bugs but are not

- **RoBERTa + `pooler_output`.** Supported by the HF model. See
  architecture notes for why I might still change it.
- **Zeros in word-level averages.** Skipped words. The sentence-level
  averager converts 0 → NaN before the 12-reader mean; the word-level
  `get_average.py` currently has that replace commented out, then
  fills NaN with 0 at the end. Different missing-data policy on
  purpose? Unclear. I would unify them on a second pass.
- **Predicted gaze means of ~21 for nFix.** Different unit system,
  see the glossary.
- **Neutral = 1, not 0.** SST-style three-way; 0 is negative so that
  the classes are ordered. `CrossEntropy` does not care about order.

---

## Experiments I would run next (personal)

1. **Fix the test loop**, create `models/`, re-run Track B once,
   keep the val curve.
2. **Per-class F1** on Track A out-of-fold predictions, especially
   class 1. If gaze only moves negative vs positive, that is a
   different paper than "gaze helps mixed reviews."
3. **Length control.** Concat `log(SentLen)` instead of the five
   gaze features. Same 16-D tower. If it matches fusion, stop.
4. **Two-feature ablation.** `nFixations + TRT` only. Then
   `GPT` only. I already suspect GPT is doing most of the work
   on contrastive reviews (sentence 4).
5. **Dropout only on the 16-D tower**, not on the 768-D pooler.
6. **Mean pool vs pooler** for RoBERTa, both with and without gaze.
7. **Word-level late-ish fusion:** mean-pool the word gaze vectors
   that align to non-punctuation tokens, instead of the
   `DataTransformer` sentence average. Same 5-D width, different
   aggregation (include skips as zeros vs exclude them).
8. **Subject-held-out gaze.** Train fusion on 11-reader means,
   test using the 12th reader's vector. That asks whether the
   model is locked to the group average. Harder, more honest.
9. **Do not predict gaze on SST until Track A is clean.** Track B
   is a force-multiplier for a signal I have not finished measuring.

---

## Environment notes for this documentation pass

- Repo: `github.com/sapphireran/Transformer-Emotion-Analysis-with-Gaze`
- Original commit: `7983303` ("first commit"), January 2024
- This environment: Python 3.12, no `pandas` / `torch` /
  `transformers`. Example scripts are stdlib-only on purpose.
- No `.mat` files, no checkpoints, no `all/*.txt` folders.
- Plots in `result/` opened and used as qualitative checks, not
  re-rendered.

## Commentary added in the Python files

The training and data-prep scripts now have English module
docstrings and section comments that point back here. Behavior is
unchanged: same defaults, same paths, same bugs. If a comment and
a docstring ever disagree, believe the code and file an update to
the comment.

## Naming I will keep using

- **Track A** = ZuCo 400, real gaze, 5-fold CV.
- **Track B** = full SST, predicted gaze, hold-out.
- **Late fusion** = 768-D pooler + 16-D gaze, one classifier.
- **Gaze tower** = `Linear(5, 16)` only.

I will not call Track B "eye-tracking sentiment analysis" in any
note that might leave this folder.
