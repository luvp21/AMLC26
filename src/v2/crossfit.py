"""K-fold cross-fitting over training-fold S1 entities (matcher and pruner).

Model g is trained on training rows whose S1 is not in group g. Training rows
of group g are scored by model g only (never by a model that saw them). Every
other row (tune, holdout, test) gets the mean of the K models, so each kind of
row gets the same kind of probability, as on test.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def cross_fit(X: np.ndarray, train_rows: np.ndarray, groups: np.ndarray, k: int,
              fit: Callable[[np.ndarray, int], object], log=print, tag: str = "") -> tuple[np.ndarray, list]:
    """fit(row_mask, g) -> model with predict_proba. Returns (p for every row, models)."""
    models = []
    for g in range(k):
        rows = train_rows & (groups != g)
        models.append(fit(rows, g))
        log(f"  {tag} group {g}: fit on {rows.sum():,} rows")
    p = np.zeros(len(X), dtype=np.float64)
    other = np.flatnonzero(~train_rows)
    if len(other):
        p[other] = mean_predict(models, X[other])
    for g, m in enumerate(models):
        rows = np.flatnonzero(train_rows & (groups == g))
        if len(rows):
            p[rows] = m.predict_proba(X[rows])[:, 1]
    return p, models


def mean_predict(models: list, X: np.ndarray, chunk: int = 5_000_000) -> np.ndarray:
    out = np.empty(len(X), dtype=np.float64)
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = np.mean([m.predict_proba(X[i:i + chunk])[:, 1] for m in models], axis=0)
    return out
