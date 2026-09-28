"""The matching classifier: given pairwise features for a (Source 1,
candidate) pair, predict match probability, then select final matches per
Source 1 entity via a probability threshold tuned to maximize macro F_0.5
directly (not left at a generic 0.5 cutoff) — F_0.5 weights precision 2x
over recall, so the right threshold is usually higher than 0.5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import entity_resolution_f_score


def train_matcher(X: np.ndarray, y: np.ndarray, seed: int = 42, model_config=None, n_jobs: int = -1):
    """Defaults reproduce the baseline model exactly; pass a
    config.ModelConfig to change anything. n_jobs caps threads (shared box)."""
    import lightgbm as lgb

    from src.entity_resolution.config import ModelConfig

    cfg = model_config or ModelConfig(seed=seed)
    model = lgb.LGBMClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        random_state=cfg.seed,
        is_unbalance=cfg.is_unbalance,
        verbosity=-1,
        n_jobs=n_jobs,
    )
    model.fit(X, y)
    return model


def select_matches(
    s1_ids: list[str], candidate_ids: list[str], probabilities: np.ndarray, threshold: float
) -> dict[str, list[str]]:
    """Group (s1_id, candidate_id, proba) triples into per-entity match lists,
    keeping only candidates above `threshold`. Entities with no candidate
    surviving the threshold get an empty list (correct format for a
    predicted singleton)."""
    predictions: dict[str, list[str]] = {}
    for s1_id, cand_id, proba in zip(s1_ids, candidate_ids, probabilities):
        predictions.setdefault(s1_id, [])
        if proba >= threshold:
            predictions[s1_id].append(cand_id)
    return predictions


def tune_threshold(
    s1_ids: list[str],
    candidate_ids: list[str],
    probabilities: np.ndarray,
    ground_truth: dict[str, set[str]],
    thresholds: tuple[float, ...] = tuple(round(t, 2) for t in np.arange(0.1, 0.96, 0.05)),
) -> tuple[float, float, dict[float, float]]:
    """Sweep thresholds, score each against `ground_truth` with the real
    macro F_0.5 metric, return (best_threshold, best_score, all_scores)."""
    all_scores = {}
    for threshold in thresholds:
        predictions = select_matches(s1_ids, candidate_ids, probabilities, threshold)
        score = entity_resolution_f_score(predictions, ground_truth)
        all_scores[threshold] = score
    best_threshold = max(all_scores, key=all_scores.get)
    return best_threshold, all_scores[best_threshold], all_scores
