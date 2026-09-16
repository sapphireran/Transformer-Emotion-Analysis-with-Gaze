import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from examples.lib.fusion import hashed_text_matrix, run_fusion_cv
from examples.lib.loaders import load_zuco_experiment


def test_hashed_text_is_deterministic():
    sentences = ["slow silly hilarious", "slow silly hilarious"]
    a = hashed_text_matrix(sentences)
    b = hashed_text_matrix(sentences)
    assert a.shape == (2, 2**12)
    assert (a - b).nnz == 0
    # Identical strings must hash to the same row.
    assert (a[0] - a[1]).nnz == 0


def test_run_fusion_cv_shapes_and_bounds():
    df = load_zuco_experiment("standard")
    results = run_fusion_cv(df, n_splits=3, random_state=0)
    for kind in ("majority", "length", "gaze", "text", "fusion"):
        assert kind in results
        assert len(results[kind].folds) == 3
        for fold in results[kind].folds:
            assert fold.metrics.n == 400 // 3 or fold.metrics.n == 400 // 3 + 1
            assert 0.0 <= fold.metrics.accuracy <= 1.0
            assert 0.0 <= fold.metrics.f1_macro <= 1.0


def test_fold_membership_matches_trainer_seed():
    """Same StratifiedKFold recipe as model_ZuCo_SST.py."""
    df = load_zuco_experiment("standard")
    y = df["sentiment_label"].astype(int).to_numpy()
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sizes = [len(test) for _, test in kf.split(df, y)]
    assert sizes == [80, 80, 80, 80, 80]
    # First fold test ids are a regression lock so a later refactor
    # cannot silently change the protocol.
    first_test = next(kf.split(df, y))[1]
    assert first_test[:5].tolist() == [2, 6, 7, 8, 9]
