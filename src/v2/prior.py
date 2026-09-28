"""Label-free prior (base-rate) correction per cell, EM of Saerens et al. (2002).

Test holds many more neighbour look-alikes than train (a same-name candidate on
the same street with a different house number is a true match 22-43% of the
time in train, far less in test). A calibrated model carries train's base rate
for such cells. For each cell (country x house-number relation) the test base
rate is estimated from the model's own calibrated probabilities on the test
pairs of that cell:

  repeat:  p'_i = r1 p_i / (r1 p_i + r0 (1 - p_i)),  r1 = pi/pi_tr, r0 = (1-pi)/(1-pi_tr)
           pi   = mean(p'_i)

and the cell's probabilities are adjusted by the resulting log-odds shift.
Train base rates per cell come from the tune fold. Sanity check: on the
holdout (train's mix) the estimated shifts should be close to 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.v2.house import REL

MIN_CELL = 2000  # cells with fewer pairs are left unadjusted
CLIP = (-6.0, 6.0)


def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def train_priors(y: np.ndarray, rel: np.ndarray, country: np.ndarray) -> dict:
    """{(country, rel): base rate} plus {("*", rel): pooled base rate}, from labelled (tune) rows."""
    df = pd.DataFrame({"y": y.astype(float), "rel": rel.astype(int), "country": country})
    out = {(c, r): v for (c, r), v in df.groupby(["country", "rel"]).y.mean().items()}
    out.update({("*", r): v for r, v in df.groupby("rel").y.mean().items()})
    return out


def em_prior(p: np.ndarray, pi_tr: float, iters: int = 100, tol: float = 1e-7) -> float:
    pi = pi_tr
    for _ in range(iters):
        r1, r0 = pi / pi_tr, (1 - pi) / (1 - pi_tr)
        adj = r1 * p / (r1 * p + r0 * (1 - p))
        new = float(adj.mean())
        if abs(new - pi) < tol:
            return new
        pi = new
    return pi


def cell_shifts(p: np.ndarray, rel: np.ndarray, country: np.ndarray, priors: dict) -> pd.DataFrame:
    """Per (country, rel): pairs, train base rate, EM test base rate, log-odds shift."""
    rows = []
    for (c, r), idx in pd.DataFrame({"c": country, "r": rel.astype(int)}).groupby(["c", "r"]).indices.items():
        pi_tr = priors.get((c, r), priors.get(("*", r)))
        if pi_tr is None or len(idx) < MIN_CELL or not 0 < pi_tr < 1:
            rows.append((c, r, len(idx), pi_tr, np.nan, 0.0))
            continue
        pi_te = em_prior(p[idx], pi_tr)
        shift = float(np.clip(logit(pi_te) - logit(pi_tr), *CLIP))
        rows.append((c, r, len(idx), pi_tr, pi_te, shift))
    df = pd.DataFrame(rows, columns=["country", "rel", "pairs", "train_rate", "em_test_rate", "shift"])
    df["relation"] = df.rel.map({v: k for k, v in REL.items()})
    return df


def apply_shifts(p: np.ndarray, rel: np.ndarray, country: np.ndarray, shifts: pd.DataFrame) -> np.ndarray:
    key = pd.MultiIndex.from_arrays([country, rel.astype(int)])
    s = shifts.set_index(["country", "rel"])["shift"].reindex(key).fillna(0.0).to_numpy()
    return 1 / (1 + np.exp(-(logit(p) + s)))


def main() -> None:
    """python -m src.v2.prior   -> OUT_DIR/model/priors.pkl (train base rates per cell, tune fold)."""
    import pickle

    from src.v2 import config as C
    from src.v2.common import labels, load_records

    out = C.tag_dir("train")
    rec = load_records("train", ["country", "fold", "owner"])
    pairs = pd.read_parquet(f"{out}/features.parquet", columns=["q", "s"])
    rel = pd.read_parquet(f"{out}/features_house.parquet", columns=["house_rel"]).house_rel.to_numpy()
    s = pairs.s.to_numpy()
    tune = rec.fold.to_numpy()[s] == "tune"
    y = labels(pairs, rec.owner.to_numpy())
    priors = train_priors(y[tune], rel[tune], rec.country.to_numpy()[s][tune])
    with open(f"{C.model_dir()}/priors.pkl", "wb") as f:
        pickle.dump(priors, f)
    for k, v in sorted(priors.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(f"  {k[0]:6s} {({v2: k2 for k2, v2 in REL.items()})[k[1]]:24s} train base rate {v:.4f}")


if __name__ == "__main__":
    main()
