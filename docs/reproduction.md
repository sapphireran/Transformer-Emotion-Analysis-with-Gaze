# Reproduction (personal, CPU-first)

## What this branch guarantees

On a machine with `pandas`, `numpy`, and `scikit-learn`:

```bash
python3 -m pip install -r requirements.txt
python3 examples/run_all.py
python3 -m unittest discover -s tests -v
```

That recomputes the atlas tables under `examples/outputs/` and checks:

- every expected CSV/PNG exists
- subject-3 packing against `SentLen` and word tokens
- full-SST id partition + the two known text leaks
- PCA: PC1 > 90% on the five train gaze columns
- fusion concat dim 784
- last-batch slice is 162 of 1186

No network, no `torch`, no `transformers`.

## What the 2024 trainers need

`requirements-train.txt` plus:

- Hugging Face access for `bert-base-uncased` / `roberta-base`
- `models/` created before the first `torch.save`
- a GPU if you actually want the full-SST batch-256 run to finish in a
  sitting (it *will* run on CPU; it will be slow)
- for `read_ZuCo_mat.py`: the missing `ZuCo_mat_data/` MATLAB bundle and a
  path separator fix

Even then, two numbers in `model_full_SST.py` should not be trusted without
a one-line patch: test metrics (last batch) and “best F1” (it is accuracy).

This branch does **not** apply those patches. The atlas is documentation of
the checkout, not a rewrite of the experiment.

## MATLAB originals

ZuCo’s `.mat` files are not in the repository. You cannot regenerate
`*_SR.csv` here. Treat the CSVs as the source of truth for this personal
clone.

## Suggested CPU checks before a GPU rerun

1. Run `examples/04_subject3_reindex.py` and decide whether to re-average
   on original ids.
2. Run `examples/03_sidecar_rank.py` and decide whether five collinear
   inputs are worth a 16-d layer.
3. Fix the test-loop `extend` before quoting a test score.
4. `mkdir -p models`.
5. Set seeds.
6. Log per-class F1, not only weighted averages.

## Environment note

This work was done in a personal Cursor Cloud checkout of
`github.com/sapphireran/Transformer-Emotion-Analysis-with-Gaze`. No
company repositories were used.
