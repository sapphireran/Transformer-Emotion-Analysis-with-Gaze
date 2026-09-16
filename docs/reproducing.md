# Reproducing locally

What you can do with **only this git clone**, versus what needs extra
artifacts.

## Always available (no extra downloads)

The CSVs and PNGs are in the tree. From the repo root:

```bash
python examples/inspect_sentence_gaze.py
python examples/inspect_word_gaze.py
python examples/subject_coverage.py
python examples/label_and_gaze_summary.py
python examples/sst_split_sanity.py
python examples/toy_fusion_forward.py
python examples/gaze_only_baseline.py
python examples/metrics_example.py
```

Need: Python 3.10+ and `numpy` (`pip install numpy` if the system package
is missing). No GPU, no Hugging Face weights, no MATLAB.

Interpretation of each script: [../examples/README.md](../examples/README.md).

## Training transformers

```bash
pip install -r requirements.txt
mkdir -p models
python model_ZuCo_SST.py      # Track A
python model_full_SST.py      # Track B
```

Need:

- PyTorch with a CUDA build if you do not want multi-hour CPU fine-tunes
- Network access to pull `bert-base-uncased` / `roberta-base`
- ~500 MB+ disk for the Hub cache
- For Track B, GPU memory for batch 256 × 128 tokens, or edit `batch_size`

First Hub download is the slow step, not the 400-sentence CV.

## Rebuilding CSVs from ZuCo MATLAB

Need the official ZuCo `.mat` release laid out as:

```text
ZuCo_mat_data/task1/<12 matlab files>
```

Then fix `get_matfiles` so `subdir` uses `os.path.join('ZuCo_mat_data')`
instead of Windows backslashes, point `read_ZuCo_mat.py` at
`ZuCo_et_csv_data/` (or mkdir `et_csv_data`), and run:

```bash
python read_ZuCo_mat.py
python get_average_sentence_level.py
```

Word-level export is the same `DataTransformer` with `level='word'` (not
wired in `read_ZuCo_mat.py` today).

## Rebuilding SST label tables

Need the `ZuCo_SST_data/all/{NEGATIVE,NEUTRAL,POSITIVE}/*.txt` tree (stems
= sentence ids). Then:

```bash
python convert_full_SST.py
```

Full SST string-label dump is already `SST_data/stts_all_sentence_level.csv`.
Placeholder word rows:

```bash
python SST_data/convert_sst_to_et.py   # run with cwd = SST_data
```

Filling those zeros with a **predictor** is out of tree; use the checked-in
`gaze_prediction/data/prediction_test_v2.csv` and `*_full_sst.csv`.

## NLTK

`convert_sst_to_et.py` calls `nltk.download('punkt')`. On a dark VM that
fails; the examples do not use NLTK.

## Suggested personal checklist before a GPU run

1. `python examples/sst_split_sanity.py` — splits disjoint, label sets {0,1,2}.
2. `python examples/gaze_only_baseline.py` — linear floor on Track A.
3. Set `model_type` to the text-only sibling of the fusion model you care
   about; run that first so fusion has a baseline from the **same** code
   path.
4. Create `models/`.
5. After Track B, **do not** quote the printed test line until the
   overwrite bug is fixed; quote validation accuracy or score saved logits
   yourself.

## Environment this docs pass was written against

- Repo: `sapphireran/Transformer-Emotion-Analysis-with-Gaze`
- Branch at time of writing: personal docs/examples expansion
- Python 3.12, numpy available, pandas **not** required for examples
