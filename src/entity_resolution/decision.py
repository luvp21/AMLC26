"""Default decision step and the experiment score S.

Decision (from Phase 2 on): every Source 1 entity competes for every
candidate record. Rows the model was trained on are scored out-of-fold
(2-fold cross-fit over training entities), everything else by the model
itself, so each pair's probability is out-of-sample, as on test. Each S2/S3
record is kept only for its highest-probability S1 entity, then a threshold
tuned on the tune fold with exclusivity on.

S = 0.383 * F_US + 0.468 * F_India + 0.150 * LOCO_avg (test country mix; LOCO
stands in for France). Deltas get a paired bootstrap over S1 entities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.entity_resolution.evaluate import f05
from src.entity_resolution.harness_io import record_exclusive

S_WEIGHTS = {"US": 0.383, "India": 0.468, "LOCO": 0.150}


def score_with_oof(X, y, s1_ids, train_mask, fit, seed: int) -> np.ndarray:
    """p for every row: rows outside train_mask from fit(all train rows);
    train rows from two models, each fit on the other half of the training
    entities. `fit(mask) -> model` so callers control caching."""
    p = np.empty(len(y), dtype=np.float64)
    rest = ~train_mask
    if rest.any():
        p[rest] = fit(train_mask).predict_proba(X[rest])[:, 1]
    train_ids = pd.unique(s1_ids[train_mask])
    half = np.random.default_rng(seed).permutation(train_ids)[: len(train_ids) // 2]
    in_a = np.isin(s1_ids, half) & train_mask
    in_b = train_mask & ~in_a
    p[in_b] = fit(in_a).predict_proba(X[in_b])[:, 1]
    p[in_a] = fit(in_b).predict_proba(X[in_a])[:, 1]
    return p


class EntityScorer:
    """Per-entity F0.5 over a fixed pair table, with the pair -> entity index
    computed once (the table can hold tens of millions of pairs)."""

    def __init__(self, entities: pd.DataFrame, pairs: pd.DataFrame):
        self.ids = entities.s1_id.to_numpy()
        self.T = entities["T"].to_numpy()
        self.pos = pd.Series(np.arange(len(entities)), index=self.ids)
        self.pair_ent = self.pos.reindex(pairs.s1_id.to_numpy()).to_numpy()
        if np.isnan(self.pair_ent).any():
            raise ValueError("pairs contain s1_ids missing from entities")
        self.pair_ent = self.pair_ent.astype(np.int64)
        self.label = pairs.label.to_numpy()

    def positions(self, ids) -> np.ndarray:
        return self.pos.loc[np.asarray(ids)].to_numpy()

    def f(self, positions: np.ndarray, pred) -> np.ndarray:
        pred = np.asarray(pred, dtype=bool)
        n = len(self.ids)
        K = np.bincount(self.pair_ent, weights=pred, minlength=n)[positions]
        TP = np.bincount(self.pair_ent, weights=pred & self.label, minlength=n)[positions]
        return f05(self.T[positions], K, TP)


def decide(scorer: EntityScorer, tune_positions, p, keep, thresholds) -> tuple[float, float]:
    """Threshold maximizing tune-fold macro F0.5 with the decision mask `keep`."""
    best_t, best_f = thresholds[0], -1.0
    for t in thresholds:
        f = scorer.f(tune_positions, (p >= t) & keep).mean()
        if f > best_f:
            best_t, best_f = t, float(f)
    return best_t, best_f


def s_components(per_entity: pd.DataFrame) -> dict[str, float]:
    """per_entity: country, f_val, f_loco (f_loco of a US entity comes from the
    India-trained model and vice versa)."""
    g = per_entity.groupby("country")
    val, loco = g.f_val.mean(), g.f_loco.mean()
    out = {"F_US": val.get("US", np.nan), "F_India": val.get("India", np.nan),
           "LOCO_US_to_India": loco.get("India", np.nan), "LOCO_India_to_US": loco.get("US", np.nan)}
    out["LOCO_avg"] = (out["LOCO_US_to_India"] + out["LOCO_India_to_US"]) / 2
    out["S"] = S_WEIGHTS["US"] * out["F_US"] + S_WEIGHTS["India"] * out["F_India"] + S_WEIGHTS["LOCO"] * out["LOCO_avg"]
    return out


def paired_bootstrap(new: pd.DataFrame, ref: pd.DataFrame, n_boot: int = 1000, seed: int = 42,
                     batch: int = 50) -> pd.DataFrame:
    """Paired bootstrap over S1 entities of the deltas (new - ref) in S and each
    component. Both frames: s1_id, country, f_val, f_loco on the same entities."""
    m = new.merge(ref, on=["s1_id", "country"], suffixes=("_new", "_ref"), validate="one_to_one")
    if len(m) != len(new) or len(m) != len(ref):
        raise ValueError("bootstrap needs the same validation entities in both experiments")
    d_val = (m.f_val_new - m.f_val_ref).to_numpy()
    d_loco = (m.f_loco_new - m.f_loco_ref).to_numpy()
    is_us = (m.country == "US").to_numpy().astype(np.float64)
    is_in = (m.country == "India").to_numpy().astype(np.float64)
    n = len(m)
    rng = np.random.default_rng(seed)
    rows = []
    for start in range(0, n_boot, batch):
        b = min(batch, n_boot - start)
        W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(b)]).astype(np.float64)
        n_us, n_in = W @ is_us, W @ is_in
        comp = {
            "F_US": (W @ (d_val * is_us)) / n_us, "F_India": (W @ (d_val * is_in)) / n_in,
            "LOCO_US_to_India": (W @ (d_loco * is_in)) / n_in, "LOCO_India_to_US": (W @ (d_loco * is_us)) / n_us,
        }
        comp["LOCO_avg"] = (comp["LOCO_US_to_India"] + comp["LOCO_India_to_US"]) / 2
        comp["S"] = S_WEIGHTS["US"] * comp["F_US"] + S_WEIGHTS["India"] * comp["F_India"] + S_WEIGHTS["LOCO"] * comp["LOCO_avg"]
        rows.append(pd.DataFrame(comp))
    boot = pd.concat(rows, ignore_index=True)
    point_new, point_ref = s_components(new), s_components(ref)
    out = pd.DataFrame({
        "delta": {k: point_new[k] - point_ref[k] for k in boot.columns},
        "ci_low": boot.quantile(0.025), "ci_high": boot.quantile(0.975),
    })
    out["significant"] = (out.ci_low > 0) | (out.ci_high < 0)
    return out
