"""Competition scoring metrics.

2026's task is business entity resolution, scored by macro-averaged F_0.5 over
Source 1 entities — see `entity_resolution_f_score` below and
docs/PROBLEM_STATEMENT.md. The smape/mape_score/exact_match_f1 functions below
are prior years' metrics (2023/2025/2024 respectively), kept only as reference
— they don't apply to this year's task.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric MAPE, as percent. Used in the 2025 pricing challenge."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def mape_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Amazon's scaled MAPE score: max(0, 100*(1-MAPE)). Higher is better.

    Used in the 2023 product-length challenge.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps)))
    return float(max(0.0, 100 * (1 - mape)))


_UNIT_ALIASES = {
    "volt": "volt", "volts": "volt", "v": "volt",
    "watt": "watt", "watts": "watt", "w": "watt",
    "gram": "gram", "grams": "gram", "g": "gram",
    "kilogram": "kilogram", "kilograms": "kilogram", "kg": "kilogram",
    "centimetre": "centimetre", "centimeter": "centimetre", "cm": "centimetre",
    "millimetre": "millimetre", "millimeter": "millimetre", "mm": "millimetre",
    "inch": "inch", "inches": "inch", "in": "inch",
    "litre": "litre", "liter": "litre", "l": "litre",
    "millilitre": "millilitre", "milliliter": "millilitre", "ml": "millilitre",
}

_VALUE_UNIT_RE = re.compile(r"^\s*([\d.]+)\s*([a-zA-Z]+)\s*$")


def normalize_unit_value(raw: str) -> tuple[float, str] | None:
    """Parse a "121 volt" style string into (121.0, "volt") on a canonical unit.

    Returns None if the string can't be parsed — callers should treat that
    as a non-match. Extend `_UNIT_ALIASES` per the 2026 entity vocabulary
    once it's published.
    """
    if not isinstance(raw, str):
        return None
    m = _VALUE_UNIT_RE.match(raw.strip().lower())
    if not m:
        return None
    value_str, unit_str = m.groups()
    unit = _UNIT_ALIASES.get(unit_str)
    if unit is None:
        return None
    try:
        return float(value_str), unit
    except ValueError:
        return None


def exact_match_f1(y_true: list[str], y_pred: list[str], value_tol: float = 1e-3) -> float:
    """F1 over predictions treated as correct iff they parse to the same
    (value, unit) as the ground truth after normalization. Mirrors the 2024
    entity-value-extraction scorer.
    """
    tp = fp = fn = 0
    for true_raw, pred_raw in zip(y_true, y_pred):
        true_parsed = normalize_unit_value(true_raw)
        pred_parsed = normalize_unit_value(pred_raw)
        if pred_parsed is None:
            if true_parsed is not None:
                fn += 1
            continue
        if true_parsed is None:
            fp += 1
            continue
        tv, tu = true_parsed
        pv, pu = pred_parsed
        if tu == pu and abs(tv - pv) <= value_tol:
            tp += 1
        else:
            fp += 1
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _f_beta_on_sets(pred_set: set[str], true_set: set[str], beta: float = 0.5) -> float:
    """F_beta for one Source 1 entity's predicted vs. true matched-ID sets.

    Matches the spec exactly: a true singleton (empty true_set) scores 1.0 for
    a correct empty prediction, 0.0 for any predicted match (false merge).
    """
    if not pred_set and not true_set:
        return 1.0
    if not pred_set or not true_set:
        return 0.0
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    beta2 = beta ** 2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def parse_id_list(raw: str | float | None) -> set[str]:
    """Parse a comma-separated `matched_entity_ids`/`candidate_entity_ids` cell
    into a set of IDs. Empty/NaN/whitespace-only -> empty set (singleton)."""
    if not isinstance(raw, str) or not raw.strip():
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def entity_resolution_f_score(
    predictions: dict[str, Iterable[str]],
    ground_truth: dict[str, Iterable[str]],
    beta: float = 0.5,
) -> float:
    """Macro-averaged F_beta (default 0.5) over every Source 1 entity in
    `ground_truth` — this year's actual scoring metric (see
    docs/PROBLEM_STATEMENT.md). `predictions`/`ground_truth` map
    source1_entity_id -> an iterable of matched entity_ids (already parsed,
    e.g. via `parse_id_list`, or a set/list). A Source 1 entity present in
    `ground_truth` but missing from `predictions` is scored as an empty
    prediction.

    Worked example from the spec: pred={S2-00047,S2-00193,S3-00812},
    true={S2-00047,S3-00812} -> precision=2/3, recall=1.0, F_0.5≈0.714.
    """
    scores = []
    for s1_id, true_ids in ground_truth.items():
        true_set = set(true_ids)
        pred_set = set(predictions.get(s1_id, ()))
        scores.append(_f_beta_on_sets(pred_set, true_set, beta))
    return float(np.mean(scores)) if scores else 0.0


def score_submission_tsv(predictions_tsv: str, ground_truth_tsv: str, beta: float = 0.5) -> float:
    """Score a `matching_results.tsv`-shaped file against a ground-truth TSV
    of the same shape (`source1_entity_id`, `matched_entity_ids` columns).
    For scoring your own held-out validation split — the real test set has no
    ground truth.
    """
    import pandas as pd

    pred_df = pd.read_csv(predictions_tsv, sep="\t", dtype=str)
    gt_df = pd.read_csv(ground_truth_tsv, sep="\t", dtype=str)
    predictions = {row.source1_entity_id: parse_id_list(row.matched_entity_ids) for row in pred_df.itertuples()}
    ground_truth = {row.source1_entity_id: parse_id_list(row.matched_entity_ids) for row in gt_df.itertuples()}
    return entity_resolution_f_score(predictions, ground_truth, beta=beta)
