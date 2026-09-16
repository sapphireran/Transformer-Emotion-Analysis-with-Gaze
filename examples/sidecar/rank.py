"""Collinearity / PCA helpers for the 5-d gaze sidecar."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def collinearity_matrix(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    return df.loc[:, list(cols)].corr()


def gaze_pca(df: pd.DataFrame, cols: Sequence[str]) -> dict:
    """Standardize columns, then SVD. Returns explained-variance ratios and loadings.

    On the committed full-SST train split the first component captures ~91.7%
    of variance and the second (almost entirely GD) another ~8.2%. The
    remaining three directions are numerical dust. A 5->16 linear sidecar
    is therefore overcomplete relative to the actual table rank.
    """
    X = df.loc[:, list(cols)].to_numpy(dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    Z = (X - mean) / std
    # economy SVD
    _, singular, vt = np.linalg.svd(Z, full_matrices=False)
    eig = singular ** 2
    explained = eig / eig.sum()
    loadings = pd.DataFrame(vt, columns=list(cols), index=[f"PC{i+1}" for i in range(len(cols))])
    return {
        "n": int(Z.shape[0]),
        "singular_values": singular,
        "explained_variance_ratio": explained,
        "cumulative": np.cumsum(explained),
        "loadings": loadings,
        "mean": mean,
        "std": std,
    }
