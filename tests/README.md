"""Tests live at the repo root.

    python3 -m unittest discover -s tests -v

They import ``examples/sidecar`` (added to ``sys.path``) and read the
committed CSVs. No Hugging Face downloads, no GPU.
