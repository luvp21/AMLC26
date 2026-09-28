"""Post-processing tricks that separated top-10 from mid-table in past editions.

- 2023 (length, scaled MAPE): log-transform + clip target, then snap predictions
  to the nearest value actually seen in training, and blend two models by
  taking the min to correct right-skew bias (greenfish8090, 2nd place 2023).
- 2024 (entity extraction, F1): normalize units before scoring — see
  `metrics.normalize_unit_value`.
- 2025 (price, SMAPE): clip to a sane train-derived range; SMAPE punishes
  relative error hardest on small values, so avoid predicting near-zero.
"""
from __future__ import annotations

import numpy as np


def log1p_transform(y: np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(y, dtype=float))


def inverse_log1p(y_log: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(y_log, dtype=float))


def clip_to_train_range(preds: np.ndarray, train_values: np.ndarray, pad: float = 0.0) -> np.ndarray:
    """Clip predictions to [min, max] of the training target, optionally padded."""
    train_values = np.asarray(train_values, dtype=float)
    lo, hi = train_values.min(), train_values.max()
    span = hi - lo
    return np.clip(preds, lo - pad * span, hi + pad * span)


def snap_to_nearest_training_value(preds: np.ndarray, train_values: np.ndarray) -> np.ndarray:
    """Snap each prediction to the nearest value actually observed in training.

    Useful when the target is effectively discrete/quantized (e.g. product
    length rounded to common packaging sizes) — a trick the 2023 runner-up used.
    """
    sorted_train = np.sort(np.unique(np.asarray(train_values, dtype=float)))
    preds = np.asarray(preds, dtype=float)
    idx = np.searchsorted(sorted_train, preds)
    idx = np.clip(idx, 1, len(sorted_train) - 1)
    left = sorted_train[idx - 1]
    right = sorted_train[idx]
    return np.where(np.abs(preds - left) <= np.abs(preds - right), left, right)


def bias_correct_min(preds_a: np.ndarray, preds_b: np.ndarray) -> np.ndarray:
    """Take the elementwise min of two model outputs.

    Used by the 2023 2nd-place team (BERT + RoBERTa) to correct systematic
    over-prediction on a right-skewed target. Only apply this after checking
    your own models actually skew high — verify on OOF predictions first.
    """
    return np.minimum(np.asarray(preds_a, dtype=float), np.asarray(preds_b, dtype=float))
