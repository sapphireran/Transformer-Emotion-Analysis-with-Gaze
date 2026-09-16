# Research notes — why this setup

Personal notes. Not a paper abstract.

## The question I actually care about

Transformers are already strong on SST-style movie reviews. The interesting
question is not "can RoBERTa classify reviews?" — it can — but:

**When a reader struggles with a sentence, does that struggle show up in
gaze in a way that is *complementary* to the words?**

Complementary is the important word. If gaze is just a noisy rewrite of
word frequency and length, the fusion head should not beat a text-only
baseline by much. If gaze captures re-reading of a sarcastic adjective or
a long go-past time on a sentiment flip ("fails to provide a reason…"),
then the 16-D gaze tower has something to do.

ZuCo is almost uniquely useful here because the same 400 review sentences
were read in an EEG/ET lab *and* they carry SST labels. That is a small
*n*, so I do not treat a 2-point accuracy bump as a result. I treat the
400-row track as a **controlled fusion test** and the 11k-row track as a
**"what if we only have predicted gaze?"** stress test.

## Two tracks, two claims

### Track A — real gaze, small *n* (`model_ZuCo_SST.py`)

- 400 sentences, 12 readers averaged.
- Gaze is measured, then z-scored.
- Evaluation is 5-fold stratified CV so every sentence is a test sentence
  once. The `train.csv` / `valid.csv` / `test.csv` files in
  `ZuCo_SST_data/` are a *different* 80/10/10 split and are **not** what
  the training script uses. I left them in place because they are handy
  for a quick sanity check (`examples/inspect_datasets.py` prints both).
- 20 epochs, batch 16, Adam 5e-5. That is a lot of epochs for 320-ish
  training sentences per fold. Overfitting is expected; the number to
  report is the **mean of the five fold scores**, not the best fold.

Claim this track can support, if the fusion model wins:

> Averaged human gaze on these 400 sentences carries a little extra
> sentiment information beyond RoBERTa's pooled representation.

Claim this track cannot support:

> Gaze features will help on arbitrary web text, or on a new reader, or
> at word level.

### Track B — predicted gaze, large *n* (`model_full_SST.py`)

- 11,853 SST sentences.
- Five gaze columns (`nFix`, `GD`, `TRT`, `FFD`, `GPT`) that do **not**
  come from people reading the full SST in a lab. They are transferred /
  predicted (see `gaze_prediction/data/` and `SST_data/convert_sst_to_et.py`).
- Hold-out 80/10/10 from `SST_data/spilt.py` (`random_state=42`).
- 5 epochs, batch 256. The script writes `models/best_{model_type}_model.pth`
  when validation **accuracy** improves.

Claim this track can support, if fusion wins:

> A gaze-shaped auxiliary vector, even a predicted one, is not harmful
> and sometimes helpful on SST.

Claim this track cannot support:

> The model "uses eye tracking." It uses a five-number summary that was
> *trained to look like* eye tracking.

That distinction is the whole reason the two scripts exist instead of one.

## Why these five features, not eight?

ZuCo sentence tables have eight numeric gaze-ish columns:

```
omissionRate, nFixations, meanPupilSize, GD, TRT, FFD, SFD, GPT
```

The model uses five: `nFixations, FFD, GPT, TRT, GD`.

Personal reasons, not a grid search I wrote down:

1. **nFix, FFD, GD, TRT, GPT** are the classic reading-time family used in
   psycholinguistics papers and in most gaze-NLP baselines. They answer
   "how often / how long / did they go back?"
2. **SFD** is undefined (or zeroed) when a word was fixated more than
   once. After sentence-level averaging it is a muddy mixture of "easy
   words" and "missing data." I would rather not feed that to a linear
   layer of size 16.
3. **omissionRate** is almost the inverse of nFixations at sentence
   level. Including both is close to duplicating a column.
4. **meanPupilSize** is interesting (arousal, load) but it is also
   lighting- and calibration-sensitive. Across 12 subjects it is a
   noisier story than reading times. I kept it in the CSV for plots,
   not in the concat.

If I rerun the ZuCo track, the first ablation is "five features vs five
plus pupil" vs "nFix + TRT only."

## Why average across readers?

Each `ZuCo_et_csv_data/{k}_SR.csv` is one person. Averaging
(`get_average_sentence_level.py`, zeros treated as missing) does three
things:

1. Shrinks subject-specific calibration error.
2. Hides reader × sentence interactions that might *be* the signal
   (some people skim sarcastic reviews; some people grind through them).
3. Gives the fusion model a single 5-D vector per sentence, which matches
   the pooled text vector (also one vector per sentence).

So the current model is **sentence-level late fusion**. It cannot say
"the model looked extra long at *failing*." That analysis lives in the
word-level CSVs and in `examples/walk_sentence_gaze.py`. A next model
could align word-level gaze to WordPiece tokens. That is a different
engineering problem (subword alignment, skipped words, punctuation).

## Why RoBERTa as the default

`model_type = 'roberta_eye_tracking'` is the default in both scripts.
RoBERTa is a familiar SST baseline and the tokenizer does not add the
`[CLS] / [SEP]` pair the same way BERT does, but the code still reads
`pooler_output` from `RobertaModel`. That works because `roberta-base`
ships a pooler; it is just less discussed than BERT's. See
`04-model-architecture.md` for the implication (the pooler is a tanh
over the first token, trained during pretraining with a different
objective than BERT's NSP).

I would not read a RoBERTa-vs-BERT gap in *this* repo as a finding about
gaze. The two checkpoints see different tokenizations of the same 400
sentences. Any comparison needs both `*_eye_tracking` and the matching
text-only run.

## What I am deliberately not doing

- **No EEG.** ZuCo has EEG. `DataTransformer` mentions it in the
  docstring. These scripts never load it.
- **No token-level loss.** Labels are sentence-level SST classes.
- **No significance tests** in the training scripts. Five CV folds are
  not enough to get cute about p-values.
- **No claim about "emotion" vs "sentiment."** The repo title says
  Emotion Analysis; the labels are SST polarity. I treat them as
  sentiment labels and keep "emotion" only as the historical folder
  name.

## Related files I treat as evidence, not as models

`result/*_scatter_hist_plots.png` are pairwise plots of predicted or
reference gaze features (PROVO, train, test). They are useful for one
sanity check: FFD, GD, TRT, GPT, nFix are **highly correlated**. A 16-D
linear projection of five correlated numbers is still basically "reading
effort." If fusion helps, it is probably helping as a single difficulty /
re-reading axis, not as five independent cognitive codes.

That is fine. A one-dimensional "this sentence was hard to read" feature
is still a legitimate auxiliary signal. I just should not write as if the
network has a dedicated FFD neuron and a dedicated GPT neuron.
