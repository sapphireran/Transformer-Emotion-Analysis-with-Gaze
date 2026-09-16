# Examples

Standard-library scripts for this personal gaze + sentiment repo. None of these import `torch`, `transformers`, `pandas`, or `sklearn`.

## Scripts

| Script | Purpose |
| --- | --- |
| `inspect_datasets.py` | row counts, columns, sample sentences |
| `schema_check.py` | required columns + numeric gaze fields |
| `label_balance.py` | class histograms and majority baselines |
| `split_audit.py` | disjoint ids, row-count additivity |
| `gaze_feature_report.py` | per-class gaze means |
| `fusion_forward_demo.py` | tiny text / gaze / fused softmax on ZuCo |
| `generate_markdown_tables.py` | refresh `outputs/dataset_inventory.md` |

```bash
python3 examples/schema_check.py
python3 examples/fusion_forward_demo.py
```

Helpers live in `examples/lib/`. Tests live in `tests/test_examples_lib.py`.

Longer context: [docs/examples.md](../docs/examples.md).
