"""Blend/stack multiple models' out-of-fold predictions. Every past edition's
top-20 (vs mid-table) came down to ensembling several model families rather
than trusting one — do this even under time pressure, it's cheap.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize


def find_blend_weights(
    oof_dict: dict[str, np.ndarray],
    y_true: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    minimize_metric: bool = True,
) -> dict[str, float]:
    """Find non-negative weights (summing to 1) that optimize `metric_fn` on
    a simple linear blend of the given OOF prediction arrays.
    """
    names = list(oof_dict.keys())
    oof_matrix = np.stack([oof_dict[n] for n in names], axis=1)  # [n_samples, n_models]
    n_models = len(names)

    def objective(w):
        w = np.abs(w) / (np.abs(w).sum() + 1e-9)
        blend = oof_matrix @ w
        score = metric_fn(y_true, blend)
        return score if minimize_metric else -score

    x0 = np.ones(n_models) / n_models
    result = minimize(objective, x0, method="Nelder-Mead")
    weights = np.abs(result.x) / (np.abs(result.x).sum() + 1e-9)
    return dict(zip(names, weights))


def apply_blend(pred_dict: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    out = np.zeros_like(next(iter(pred_dict.values())), dtype=float)
    for name, w in weights.items():
        out += w * pred_dict[name]
    return out


def stack_with_ridge(
    oof_dict: dict[str, np.ndarray],
    y_true: np.ndarray,
    test_dict: dict[str, np.ndarray],
    alpha: float = 1.0,
) -> np.ndarray:
    """Fit a Ridge meta-model on OOF predictions as features, apply to test
    predictions. Usually a small but free improvement over a fixed-weight
    blend once you have 3+ base models.
    """
    from sklearn.linear_model import Ridge

    names = list(oof_dict.keys())
    X_meta_train = np.stack([oof_dict[n] for n in names], axis=1)
    X_meta_test = np.stack([test_dict[n] for n in names], axis=1)

    meta = Ridge(alpha=alpha, positive=True)
    meta.fit(X_meta_train, y_true)
    return meta.predict(X_meta_test)
