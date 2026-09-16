#!/usr/bin/env python3
"""5-fold logistic baselines: majority, length, gaze, gaze+length, residualized gaze."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))

from gazekit.baselines import (
    fit_gaze_only,
    fit_gaze_plus_length,
    fit_length_only,
    fit_logistic,
    fit_majority,
    residualize,
)
from gazekit.io import load_sentence_table
from gazekit.paths import default_paths
from gazekit.report import format_baseline, section
from gazekit.schema import CANONICAL_GAZE


def build_report(root: Path | None = None, n_splits: int = 5, seed: int = 42) -> dict:
    paths = default_paths(root)
    df = load_sentence_table(paths.zuco_combined_standard)
    residual = residualize(df, CANONICAL_GAZE, on="n_tokens")
    reports = {
        "majority": fit_majority(df, n_splits=n_splits, seed=seed),
        "length": fit_length_only(df, n_splits=n_splits, seed=seed),
        "gaze": fit_gaze_only(df, n_splits=n_splits, seed=seed),
        "gaze_plus_length": fit_gaze_plus_length(df, n_splits=n_splits, seed=seed),
        "gaze_residualized_on_length": fit_logistic(
            residual, CANONICAL_GAZE, "gaze_residualized_on_length", n_splits=n_splits, seed=seed
        ),
    }
    return {"n": len(df), "n_splits": n_splits, "seed": seed, "reports": reports}


def render(bundle: dict) -> str:
    chunks = [
        section(
            "Gaze-only baselines on ZuCo SST (standard-scaled)",
            f"n={bundle['n']}  folds={bundle['n_splits']}  seed={bundle['seed']}\n"
            "Metrics are the same weighted trio as model_ZuCo_SST.py, plus macro-F1.\n"
            "A StandardScaler is fit *inside* each fold (stricter than the historical CSVs).",
        )
    ]
    for key in (
        "majority",
        "length",
        "gaze",
        "gaze_plus_length",
        "gaze_residualized_on_length",
    ):
        chunks.append(section(key, format_baseline(bundle["reports"][key])))
    return "\n".join(chunks)


def main() -> None:
    print(render(build_report()))


if __name__ == "__main__":
    main()
