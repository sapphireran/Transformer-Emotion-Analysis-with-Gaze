#!/usr/bin/env bash
# Run every documented example from the repository root.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH

echo "==> inspect_datasets"
python3 examples/inspect_datasets.py
echo
echo "==> gaze_feature_tour"
python3 examples/gaze_feature_tour.py
echo
echo "==> label_and_split_audit"
python3 examples/label_and_split_audit.py
echo
echo "==> fusion_architecture_demo"
python3 examples/fusion_architecture_demo.py
echo
echo "==> synthetic_training_loop"
python3 examples/synthetic_training_loop.py
echo
echo "==> pytest"
python3 -m pytest tests -q
echo
echo "All examples and tests finished."
