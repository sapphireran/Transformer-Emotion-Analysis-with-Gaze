# Known issues

Personal punch list. None of these are fixed in this docs/examples
branch; they are written down so a later training run does not surprise
you.

## Training scripts

1. **Full-SST test loop overwrites predictions.**
   `model_full_SST.py` assigns `all_preds = preds.cpu().numpy()` instead
   of extending. Only the last test batch is scored. Validation uses
   `.extend()` correctly.
2. **Best-checkpoint comment is wrong.**
   The full-SST loop tracks validation **accuracy** and prints it as F1.
3. **Hardcoded 3-class loss reshape.**
   Fusion loss is `logits.view(-1, 3)` even though `num_labels` exists.
4. **No PyTorch seed.**
   Only the ZuCo `StratifiedKFold` has `random_state=42`. GPU reruns
   of the same `model_type` are not comparable.
5. **`models/` is not created.**
   `torch.save` to `models/best_{model_type}_model.pth` fails if the
   directory is missing. `mkdir -p models` first.
6. **ZuCo track never checkpoints.**
   5 × 20 epochs of weights are thrown away after each fold.

## Paths and names

7. **Windows MATLAB path.**
   `utils_ZuCo.get_matfiles()` joins `os.getcwd()` with
   `\\ZuCo_mat_data\\`. Linux regenerations must change that string.
8. **`et_csv_data` vs `ZuCo_et_csv_data`.**
   `get_average_sentence_level.py` still points at the old folder name.
9. **`spilt.py`.**
   Both SST folders use that typo. The examples never import those
   files; they read the committed split CSVs.
10. **Cwd-sensitive helpers.**
    Several conversion scripts only work if you `cd` into their folder.
    See [reproduction.md](reproduction.md).

## Data

11. **Subject 3 is short and reindexed.**
    `3_SR.csv` has 299 rows, `id` 0–298. Averaging is by DataFrame
    index, so that reader only influences sentences 0–298. Documented
    in [data-pipeline.md](data-pipeline.md); `inspect_datasets.py`
    treats it as expected.
12. **Scaler fit on all ZuCo sentences.**
    Standard / min-max averages were fit on the full 400 rows, then
    5-fold CV reads those already-scaled values. Mild leakage into
    fold test features.
13. **Projected full-SST gaze is collinear.**
    Five columns with `|r|` often above 0.98. Significant `f_classif`
    does not mean five independent measures.
14. **Tokenizer mismatch.**
    `convert_sst_to_et.py` keeps `[A-Za-z]+` via NLTK.
    Training uses BERT / RoBERTa tokenization at `max_length=128`.
15. **Join script was never committed.**
    `examples/join_zuco_sst.py` reconstructs it. Extra `SentLen` / `id`
    columns appear on the rebuilt frame and are expected.
16. **`prediction_test_v2.csv` has 11 empty `word` fields.**
    Inventory reports them; the word-level example still runs.

## Examples (already handled)

- sklearn ≥ 1.5 dropped `LogisticRegression(multi_class=...)`.
  `sentiment_baselines.py` does not pass that argument.
- `inspect_datasets.py` no longer fails solely because subject 3 is
  short.

When you fix an item above, delete it from this list and mention the
fix in [architecture.md](architecture.md) or [data-pipeline.md](data-pipeline.md).
