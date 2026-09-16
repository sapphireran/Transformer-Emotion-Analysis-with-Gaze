# Examples

Small scripts that document the personal gaze + sentiment repo **without**
downloading BERT/RoBERTa. They only need the stdlib and `numpy`.

Run them from the repository root or from this folder:

```bash
python3 -m pip install -r requirements-examples.txt
python3 examples/run_all.py
python3 -m unittest discover -s examples/tests -v
```

| Script | What it checks |
| --- | --- |
| `inspect_datasets.py` | Row counts, columns, label mixes; compares to `docs/datasets.md` |
| `split_integrity.py` | Train/valid/test are a disjoint cover of each combined table |
| `scaling_check.py` | Rebuilt min-max / z-score match the checked-in average tables |
| `fusion_forward.py` | Numpy clone of `EyeTrackingModel` shapes and gaze sensitivity |
| `gaze_by_sentiment.py` | Class-conditional means on ZuCo gold vs full-SST transferred gaze |
| `word_to_sentence.py` | Subject-1 word means reconstruct sentence-level `nFixations` |

Helpers: `paths.py` (all CSV locations), `csvutil.py` (DictReader utilities).

None of these scripts call social APIs or load company code. They stay inside
this personal checkout.
