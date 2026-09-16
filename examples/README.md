# Personal example suite

Run from the repository root. NumPy only.

```bash
python3 examples/run_all.py
```

Each script writes a short markdown snapshot under `output/` and exits
non-zero if a locked measurement drifted (reader-3 contamination counts,
scaler fingerprints, split mixes, …).

`08_cpu_baselines.py` is the slowest step: it fits ridge probes on the
400-row ZuCo table and the 11,853-row SST table. Still seconds, not a
GPU job.

Do not import `model_ZuCo_SST.py` / `model_full_SST.py` from these
scripts — those pull Hugging Face weights at import time.
