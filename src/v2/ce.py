"""Cross-encoder rescoring of uncertain pairs (time-boxed experiment on top of m3_syn1).

  python -m src.v2.ce band  --tag aug    # m3_syn1 probabilities (out-of-fold on the training fold),
                                         # band dump + cross-encoder training sample   (CPU)
  python -m src.v2.ce band  --tag test   # test probabilities (as predict.py) + band      (CPU)
  python -m src.v2.ce train --deadline 17:00   # fine-tune (GPU, fp16 autocast + GradScaler)
  python -m src.v2.ce score --tag aug    # cross-encoder logits for the tune + holdout band (GPU)
  python -m src.v2.ce score --tag test
  python -m src.v2.ce blend              # tune-band logistic blend, decoding tuned on tune,
                                         # holdout vs GBM-only + paired bootstrap       (CPU)
  python -m src.v2.ce apply              # test: blend + decoding -> runs/submissions/m3_ce/ (CPU)

Band: pairs whose calibrated m3_syn1 probability is in [BAND_LO, BAND_HI]; pairs
outside keep the GBM probability. Text is raw "name | address" (not transliterated).
Synthetic look-alikes (aug) copy their source's raw text, so the same edit (house
digits, appended sibling word) is re-applied to it before the text is used.
Base model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0),
loaded with a fresh 1-logit classification head; run with HF_HUB_OFFLINE=1.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import time
from datetime import datetime

import numpy as np
import pandas as pd

from src.v2 import config as C
from src.v2.common import labels, load_records
from src.v2.crossfit import mean_predict
from src.v2.crowd import CROWD_FEATURES
from src.v2.house import HOUSE_FEATURES
from src.v2.novel import NOVEL_FEATURES
from src.v2.sibling import second_pass_features
from src.v2.synth import _DIGITS, replace_first
from src.v2.train import add_crowd_features, add_house_features, add_novel_features

CE_MODEL = os.environ.get("V2_CE_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
MATCHER = f"{C.MODEL_DIR}/matcher_m3_gap.pkl"
CE_DIR = f"{C.MODEL_DIR}/ce"  # per name: model/<name>/


def ce_dir(name: str) -> str:
    return f"{C.MODEL_DIR}/{name}"


def suffix(name: str) -> str:
    return "" if name == "ce" else f"_{name}"


def bsuf(band_name: str) -> str:
    return f"_{band_name}" if band_name else ""
BAND_LO, BAND_HI = 0.02, 0.98
TRAIN_CAP = 500_000
EASY_SHARE = 0.20  # share of the training sample drawn from confident (out-of-band) pairs
MAX_LEN, BATCH, LR, WARMUP = 128, 64, 3e-5, 0.05
SCORE_BATCH = 512
DEV = os.environ.get("V2_CE_DEVICE", "cuda")
AMP = DEV == "cuda"
LOG_EVERY = 200
REC_COLS = ["entity_id", "source", "country", "fold", "xgroup", "owner", "name", "name_core", "addr", "house_num",
            "postcode", "business_name", "business_address"]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- band (CPU)

def oof_predict(models: list, X: np.ndarray, train_rows: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """As in cross_fit: training rows of group g scored by model g only, all other rows by the mean."""
    p = np.empty(len(X))
    other = np.flatnonzero(~train_rows)
    if len(other):
        p[other] = mean_predict(models, X[other])
    for g, m in enumerate(models):
        rows = np.flatnonzero(train_rows & (groups == g))
        if len(rows):
            p[rows] = m.predict_proba(X[rows])[:, 1]
    return p


def gbm_scores(tag: str, m: dict, rec: pd.DataFrame, workers: int, log) -> pd.DataFrame:
    """q, s, p_raw (second pass), p (calibrated) for every pair of the tag, as the matcher saw them."""
    out = C.tag_dir(tag)
    feats = pd.read_parquet(f"{out}/features.parquet")
    if set(m["features"]) & set(HOUSE_FEATURES):
        feats = add_house_features(feats, out)
    if set(m["features"]) & set(NOVEL_FEATURES):
        feats = add_novel_features(feats, out)
    if set(m["features"]) & set(CROWD_FEATURES):
        feats = add_crowd_features(feats, out)
    q, s = feats.q.to_numpy(), feats.s.to_numpy()
    X = feats[m["features"]].to_numpy(np.float32)
    del feats
    train_rows = (rec.fold.to_numpy()[s] == "train") if tag != "test" else np.zeros(len(q), bool)
    groups = rec.xgroup.to_numpy()[s]
    t0 = time.time()
    p1 = oof_predict(m["models"], X, train_rows, groups)
    log(f"first pass scored in {time.time() - t0:.0f}s")
    X2 = np.hstack([X, second_pass_features(q, s, p1, rec, workers)])
    del X
    p_raw = oof_predict(m["models2"], X2, train_rows, groups) if m.get("models2") else p1
    log(f"second pass scored in {time.time() - t0:.0f}s")
    p = m["calibration"].predict(p_raw)
    return pd.DataFrame({"q": q.astype(np.int32), "s": s.astype(np.int32), "p_raw": p_raw.astype(np.float32),
                         "p": p.astype(np.float32)})


def raw_texts(rec: pd.DataFrame) -> np.ndarray:
    """'name | address' from the raw columns; synthetic look-alikes get their source's edit re-applied."""
    name, addr = rec.business_name.to_numpy().copy(), rec.business_address.to_numpy().copy()
    if "synthetic" in rec and rec.synthetic.any():
        syn = rec.synthetic.to_numpy()
        real = rec[~syn]
        key = real.source.astype(str) + "\t" + real.business_name + "\t" + real.business_address
        orig = pd.Series(real.index.to_numpy(), index=key.to_numpy())
        orig = orig[~orig.index.duplicated()]
        core, house = rec.name_core.to_numpy(), rec.house_num.to_numpy()
        for r in np.flatnonzero(syn):
            o = orig.get(f"{rec.source.iat[r]}\t{name[r]}\t{addr[r]}")
            if o is None:
                continue
            d_old, d_new = _DIGITS.search(house[o] or ""), _DIGITS.search(house[r] or "")
            if d_old and d_new:
                addr[r] = replace_first(addr[r], d_old.group(0), d_new.group(0))
            extra = [w for w in core[r].split() if w not in set(core[o].split())]
            if extra:
                name[r] = f"{name[r]} {' '.join(extra)}"
    return np.array([f"{n} | {a}" for n, a in zip(name, addr)], dtype=object)


def band_main(a) -> None:
    out = C.tag_dir(a.tag)
    log = C.log_to(f"{out}/ce.log")
    t0 = C.stage_start(log, f"ce band {a.tag}")
    with open(a.matcher, "rb") as f:
        m = pickle.load(f)
    rec = load_records(a.tag, REC_COLS + (["synthetic"] if a.tag == "aug" else []))
    if a.reuse and os.path.exists(f"{out}/ce_gbm.parquet"):
        g = pd.read_parquet(f"{out}/ce_gbm.parquet")
        log("reusing GBM scores from ce_gbm.parquet")
    else:
        g = gbm_scores(a.tag, m, rec, a.workers, log)
        g.to_parquet(f"{out}/ce_gbm.parquet", index=False)
    q, s, p = g.q.to_numpy(), g.s.to_numpy(), g.p.to_numpy()
    fold, country = rec.fold.to_numpy()[s], rec.country.to_numpy()[s]
    y = labels(g, rec.owner.to_numpy()) if a.tag != "test" else np.zeros(len(g), bool)
    band = (p >= a.band_lo) & (p <= a.band_hi)
    text = raw_texts(rec)
    rows = np.flatnonzero(band)
    df = pd.DataFrame({"row": rows.astype(np.int64), "q": q[rows], "s": s[rows], "label": y[rows], "fold": fold[rows],
                       "country": country[rows], "text_a": text[s[rows]], "text_b": text[q[rows]]})
    if not a.reuse or a.band_name:
        df.to_parquet(f"{out}/ce_band{bsuf(a.band_name)}.parquet", index=False)
    log(f"band [{a.band_lo}, {a.band_hi}]: {band.sum():,} of {len(g):,} pairs ({band.mean():.2%})")
    by = pd.DataFrame({"fold": fold, "country": country, "band": band, "y": y})
    log(by[by.band].groupby(["fold", "country"]).y.agg(pairs="size", positive_rate="mean").round(4).to_string())
    if a.tag != "test" and not a.no_train_sample:
        rng = np.random.default_rng(a.seed)
        tr = fold == "train"
        b_rows, e_rows = np.flatnonzero(band & tr), np.flatnonzero(~band & tr)
        budget = int(a.cap * (1 - EASY_SHARE))
        if len(b_rows) > budget and a.sample == "natural":  # uniform: the band's own positive rate
            b_rows = rng.choice(b_rows, budget, replace=False)
        elif len(b_rows) > budget:  # keep every band positive, sample band negatives
            pos, neg = b_rows[y[b_rows]], b_rows[~y[b_rows]]
            b_rows = np.r_[pos[:budget], rng.choice(neg, max(budget - len(pos), 0), replace=False)]
        n_easy = int(len(b_rows) * EASY_SHARE / (1 - EASY_SHARE))
        e_rows = rng.choice(e_rows, min(n_easy, len(e_rows)), replace=False)
        pick = rng.permutation(np.r_[b_rows, e_rows])
        ts = pd.DataFrame({"text_a": text[s[pick]], "text_b": text[q[pick]], "label": y[pick].astype(np.float32)})
        ts.to_parquet(f"{out}/ce_train{suffix(a.ce_name)}.parquet", index=False)
        log(f"cross-encoder training sample: {len(ts):,} pairs (band {len(b_rows):,}, easy {len(e_rows):,}), "
            f"positive rate {ts.label.mean():.3f}")
    C.stage_end(log, f"ce band {a.tag}", t0)


# ---------------------------------------------------------------- train / score (GPU)

def _torch():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if DEV == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    return torch, AutoTokenizer, AutoModelForSequenceClassification


def _deadline(hhmm: str) -> float:
    h, mnt = (int(x) for x in hhmm.split(":"))
    return datetime.now().replace(hour=h, minute=mnt, second=0, microsecond=0).timestamp()


def _save(model, tok, path: str, info: dict) -> None:
    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    tok.save_pretrained(tmp)
    with open(f"{tmp}/progress.json", "w") as f:
        json.dump(info, f)
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)


def train_main(a) -> None:
    torch, AutoTokenizer, AutoModel = _torch()
    from transformers import get_linear_schedule_with_warmup
    log = C.log_to(f"{C.tag_dir(a.tag)}/ce_train.log")
    df = pd.read_parquet(f"{C.tag_dir(a.tag)}/ce_train{suffix(a.ce_name)}.parquet")
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(CE_MODEL)
    model = AutoModel.from_pretrained(CE_MODEL, num_labels=1).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = math.ceil(len(df) / BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    ckpt_every = min(20_000, max(steps // 4, 1))
    deadline = _deadline(a.deadline) if a.deadline else float("inf")
    ta, tb, yy = df.text_a.tolist(), df.text_b.tolist(), df.label.to_numpy(np.float32)
    if a.swap:  # candidate text first (a different view of the pair for ensemble diversity)
        ta, tb = tb, ta
    log(f"fine-tuning {CE_MODEL}: {len(df):,} pairs, {steps:,} steps of {BATCH}, lr {LR}, warmup {WARMUP:.0%}, "
        f"fp16 autocast, checkpoint every {ckpt_every:,} steps, hard stop {a.deadline or 'none'}")
    model.train()
    t0, run_loss = time.time(), 0.0
    for step in range(steps):
        lo, hi = step * BATCH, min((step + 1) * BATCH, len(df))
        enc = tok(ta[lo:hi], tb[lo:hi], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(DEV)
        target = torch.from_numpy(yy[lo:hi]).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            logits = model(**enc).logits.squeeze(-1)
        loss = loss_fn(logits.float(), target)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        run_loss += loss.item()
        done = step + 1
        if done % LOG_EVERY == 0 or done == steps:
            el = time.time() - t0
            log(f"  step {done:,}/{steps:,} ({done / steps:.0%}) loss {run_loss / LOG_EVERY:.4f} "
                f"{hi / el:,.0f} samples/s, eta {(steps - done) * el / done / 60:.0f} min")
            run_loss = 0.0
        if done % ckpt_every == 0 and done < steps:
            _save(model, tok, f"{ce_dir(a.ce_name)}/ckpt", {"step": done, "steps": steps, "fraction": done / steps})
            log(f"  checkpoint at step {done:,}")
        if time.time() >= deadline and done < steps:
            if done / steps >= 0.5:
                _save(model, tok, f"{ce_dir(a.ce_name)}/final", {"step": done, "steps": steps, "fraction": done / steps})
                log(f"HARD STOP {a.deadline}: saved at {done / steps:.0%} of the epoch -> {ce_dir(a.ce_name)}/final")
            else:
                log(f"HARD STOP {a.deadline}: only {done / steps:.0%} of the epoch; abandoned (nothing saved as final)")
            return
    _save(model, tok, f"{ce_dir(a.ce_name)}/final", {"step": steps, "steps": steps, "fraction": 1.0})
    log(f"finished one epoch in {(time.time() - t0) / 60:.1f} min -> {ce_dir(a.ce_name)}/final")


def score_main(a) -> None:
    torch, AutoTokenizer, AutoModel = _torch()
    out = C.tag_dir(a.tag)
    log = C.log_to(f"{out}/ce.log")
    band = pd.read_parquet(f"{out}/ce_band{bsuf(a.band_name)}.parquet", columns=["row", "fold", "text_a", "text_b"])
    if a.tag != "test":  # the blend is fitted on tune and judged on holdout: training-fold band pairs are not needed
        band = band[band.fold != "train"].reset_index(drop=True)
    done = None
    base_scores = f"{out}/ce_scores{suffix(a.ce_name)}.parquet"
    if a.band_name and os.path.exists(base_scores):  # rows of the default band keep their logits
        done = pd.read_parquet(base_scores)
        known = np.isin(band.row.to_numpy(), done.row.to_numpy())
        log(f"band {a.band_name}: {known.sum():,} rows already scored, {(~known).sum():,} new")
        band = band[~known].reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(f"{ce_dir(a.ce_name)}/final")
    model = AutoModel.from_pretrained(f"{ce_dir(a.ce_name)}/final").to(DEV).eval()
    ta, tb = band.text_a.to_numpy(), band.text_b.to_numpy()
    if a.swap:  # must match how the model was trained
        ta, tb = tb, ta
    order = np.argsort([len(x) + len(y) for x, y in zip(ta, tb)])  # length-sorted batches: less padding
    z = np.empty(len(band), np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(order), SCORE_BATCH):
            idx = order[i:i + SCORE_BATCH]
            enc = tok(list(ta[idx]), list(tb[idx]), truncation=True, max_length=MAX_LEN, padding=True,
                      return_tensors="pt").to(DEV)
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                z[idx] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
            if (i // SCORE_BATCH) % 500 == 0:
                log(f"  scored {i + len(idx):,}/{len(order):,}, {(i + len(idx)) / (time.time() - t0):,.0f} pairs/s")
    res = pd.DataFrame({"row": band.row.to_numpy(), "ce_logit": z})
    if done is not None:
        res = pd.concat([done, res], ignore_index=True)
    res.to_parquet(f"{out}/ce_scores{suffix(a.ce_name)}{bsuf(a.band_name)}.parquet",
                                                                        index=False)
    log(f"cross-encoder scores for {len(z):,} new band pairs in {(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------- blend / apply (CPU)

def load_scores(out: str, names: list[str], band_name: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Band rows scored by every named cross-encoder and their logits (rows x names)."""
    frames = [pd.read_parquet(f"{out}/ce_scores{suffix(n)}{bsuf(band_name)}.parquet").rename(columns={"ce_logit": n}) for n in names]
    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on="row", how="inner")
    return df.row.to_numpy(), df[names].to_numpy(np.float64)


def blended(g: pd.DataFrame, band_rows: np.ndarray, z_ce: np.ndarray, lr) -> np.ndarray:
    """GBM calibrated p everywhere; logistic blend of [logit p, CE logit(s)] inside the band."""
    p = g.p.to_numpy().astype(np.float64).copy()
    X = np.c_[logit(p[band_rows]), z_ce]
    p[band_rows] = lr.predict_proba(X)[:, 1]
    return p


def blend_main(a) -> None:
    from sklearn.linear_model import LogisticRegression

    from src.entity_resolution.decision import EntityScorer, paired_bootstrap
    from src.entity_resolution.evaluate import error_decomposition
    from src.v2.decode import blocking_misses, fit_calibration, one_owner
    from src.v2.decode import predict as predict_rule
    from src.v2.decode import tune as tune_rules
    out = C.tag_dir(a.tag)
    log = C.log_to(f"{out}/ce_blend.log")
    rec = load_records(a.tag, ["source", "country", "fold", "owner"])
    g = pd.read_parquet(f"{out}/ce_gbm.parquet")
    names = a.ce_names.split(",")
    rows, z = load_scores(out, names, a.band_name)  # tune + holdout band pairs
    q, s = g.q.to_numpy(), g.s.to_numpy()
    owner, fold, country = rec.owner.to_numpy(), rec.fold.to_numpy(), rec.country.to_numpy()
    y = labels(g, owner)
    tune_band = fold[s[rows]] == "tune"
    lp = logit(g.p.to_numpy()[rows])

    def fit(cols):
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(np.c_[lp[tune_band], z[tune_band][:, cols]], y[rows][tune_band])
        log(f"blend {[names[c] for c in cols]} on {tune_band.sum():,} tune band pairs: coef [logit p_gbm, ...] = "
            f"{lr.coef_[0].round(3)}, intercept {lr.intercept_[0]:.3f}")
        return lr

    lr = fit(list(range(len(names))))
    p_new = blended(g, rows, z, lr)
    lr1 = fit(list(range(len(names) - 1))) if len(names) > 1 else None  # without the last cross-encoder
    ref = None
    if a.band_name:  # the same cross-encoders on the default band, for a paired comparison
        rows0, z0 = load_scores(out, names)
        tb0 = fold[s[rows0]] == "tune"
        lr0 = LogisticRegression(C=1.0, max_iter=1000).fit(np.c_[logit(g.p.to_numpy()[rows0][tb0]), z0[tb0]], y[rows0][tb0])
        ref = (rows0, z0, lr0)

    s1 = np.flatnonzero((rec.source == 1).to_numpy())
    T = np.bincount(owner[owner >= 0], minlength=len(rec))
    ents = pd.DataFrame({"s1_id": s1, "country": country[s1], "T": T[s1]})
    scorer = EntityScorer(ents, pd.DataFrame({"s1_id": s, "label": y}))
    tune = ents[fold[ents.s1_id] == "tune"]
    hold = ents[fold[ents.s1_id] == "holdout"].reset_index(drop=True)
    tune_rows = fold[s] == "tune"

    def decode(p_in, tie):
        iso = fit_calibration(p_in[tune_rows], y[tune_rows])
        p_cal = iso.predict(p_in)
        keep = one_owner(q, p_in + 1e-9 * tie)
        m = blocking_misses(T, tune.s1_id.to_numpy(), s, y)
        ch = tune_rules(scorer, scorer.positions(tune.s1_id), s, p_cal, keep, m, ("threshold", "expected_f"))
        pred = predict_rule(ch["rule"], ch["param"], s, p_cal, keep, m)
        return {"iso": iso, "m": m, "pred": pred, **ch}

    base = decode(g.p_raw.to_numpy().astype(np.float64), g.p_raw.to_numpy())
    new = decode(p_new, g.p_raw.to_numpy())
    pos = scorer.positions(hold.s1_id)
    runs = [("gbm", base), ("blend", new)]
    if lr1 is not None:
        runs.append(("blend_without_last", decode(blended(g, rows, z[:, :-1], lr1), g.p_raw.to_numpy())))
    if ref is not None:
        runs.append(("blend_default_band", decode(blended(g, ref[0], ref[1], ref[2]), g.p_raw.to_numpy())))
    per = {k: hold[["s1_id", "country"]].assign(f_val=scorer.f(pos, d["pred"]), f_loco=np.nan) for k, d in runs}
    for k, d in runs:
        f = per[k].groupby("country").f_val.mean()
        log(f"{k}: rule {d['rule']} param {d['param']}; holdout F0.5 " + ", ".join(f"{c} {v:.4f}" for c, v in f.items()))
    hold_rows = np.isin(s, hold.s1_id.to_numpy())
    edges = [0.0, 0.0005, 0.002, 0.005, 0.02, 0.98, 0.995, 1.0001]
    pg = g.p.to_numpy()
    for kind, mask in (("wrong rejects (true, not predicted)", hold_rows & y & ~new["pred"]),
                       ("wrong accepts (predicted, not true)", hold_rows & ~y & new["pred"])):
        cnt = pd.cut(pd.Series(pg[mask]), edges, right=False).value_counts(sort=False)
        log(f"holdout {kind}: {mask.sum():,}; by calibrated GBM p:\n" + cnt.to_string())
    if a.dump_pred:
        pd.DataFrame({"q": q[hold_rows & new["pred"]], "s": s[hold_rows & new["pred"]]}).to_parquet(
            f"{out}/ce_hold_pred.parquet", index=False)
        log(f"holdout predictions -> {out}/ce_hold_pred.parquet")
    hp = pd.DataFrame({"s1_id": s[hold_rows], "label": y[hold_rows]})
    log("blend error decomposition (holdout):\n" + error_decomposition(hold, hp, new["pred"][hold_rows]).round(4).to_string())
    boot = paired_bootstrap(per["blend"], per["gbm"])
    log(f"paired bootstrap, blend {names} - gbm (holdout):\n" + boot.round(4).to_string())
    if ref is not None:
        bootb = paired_bootstrap(per["blend"], per["blend_default_band"])
        log(f"paired bootstrap, band {a.band_name} - default band, blend {names} (holdout):\n" + bootb.round(4).to_string())
    if lr1 is not None:
        boot1 = paired_bootstrap(per["blend"], per["blend_without_last"])
        log(f"paired bootstrap, blend {names} - blend {names[:-1]} (holdout):\n" + boot1.round(4).to_string())
    path = f"{C.MODEL_DIR}/ce_blend{'' if names == ['ce'] else '_' + '_'.join(names)}{bsuf(a.band_name)}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"lr": lr, "iso": new["iso"], "rule": new["rule"], "param": new["param"], "m": new["m"],
                     "band": (a.band_lo, a.band_hi), "names": names, "band_name": a.band_name}, f)
    log(f"saved {path}")


def apply_main(a) -> None:
    from src.entity_resolution.evaluate import run_validator
    from src.v2.decode import one_owner
    from src.v2.decode import predict as predict_rule
    from src.v2.predict import write_tsv
    out = C.tag_dir("test")
    log = C.log_to(f"{out}/ce.log")
    with open(a.matcher, "rb") as f:
        m = pickle.load(f)
    names = a.ce_names.split(",")
    with open(f"{C.MODEL_DIR}/ce_blend{'' if names == ['ce'] else '_' + '_'.join(names)}{bsuf(a.band_name)}.pkl", "rb") as f:
        b = pickle.load(f)
    rec = load_records("test", ["entity_id", "source", "country"])
    g = pd.read_parquet(f"{out}/ce_gbm.parquet")
    rows, z = load_scores(out, b.get("names", ["ce"]), b.get("band_name", ""))
    q, s = g.q.to_numpy(), g.s.to_numpy()
    z_raw = logit(g.p_raw.to_numpy().astype(np.float64))
    if a.lookalike_shift:
        from src.v2.predict import lookalike_mask
        la = lookalike_mask(g[["q", "s"]], out)
        z_raw = np.where(la, z_raw - a.lookalike_shift, z_raw)
        log(f"look-alike shift {a.lookalike_shift} on {la.mean():.2%} of pairs")
    if a.unseen_shift:
        unseen = ~np.isin(rec.country.to_numpy()[s], m["train_countries"])
        z_raw = np.where(unseen, z_raw - a.unseen_shift, z_raw)
        log(f"unseen-country shift {a.unseen_shift} on {unseen.mean():.2%} of pairs")
    p_raw = 1 / (1 + np.exp(-z_raw))
    g = g.assign(p=m["calibration"].predict(p_raw))  # as predict.py: shifts on the raw logit, then calibration
    p_new = blended(g, rows, z, b["lr"])
    p_cal = b["iso"].predict(p_new)
    keep = one_owner(q, p_new + 1e-9 * p_raw)
    pred = predict_rule(b["rule"], b["param"], s, p_cal, keep, b["m"])
    ids = rec.entity_id.to_numpy()
    s1 = rec[rec.source == 1]
    s1_sorted = np.sort(s1.entity_id.to_numpy())
    pairs = pd.DataFrame({"s1": ids[s], "cand": ids[q]})
    os.makedirs(a.out_dir, exist_ok=True)
    write_tsv(f"{a.out_dir}/candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids", s1_sorted,
              pairs.groupby("s1").cand.agg(list))
    write_tsv(f"{a.out_dir}/matching_results.tsv", "source1_entity_id\tmatched_entity_ids", s1_sorted,
              pairs[pred].groupby("s1").cand.agg(list))
    ok, report = run_validator(f"{a.out_dir}/matching_results.tsv", f"{a.out_dir}/candidate_pairs.tsv",
                               f"{C.DATA_ROOT}/test", check_ids=True)
    log(f"validator passed: {ok}\n{report.strip()}")
    if ok:
        dest = f"runs/submissions/{a.experiment}"
        os.makedirs(dest, exist_ok=True)
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            shutil.copy2(f"{a.out_dir}/{name}", f"{dest}/{name}")
        log(f"validated files copied to {dest}/")
    country = rec.country.to_numpy()
    rows_out = []
    src1 = rec.source.to_numpy() == 1
    for c in sorted(s1.country.unique()):
        s1_c = np.flatnonzero(src1 & (country == c))
        k = np.bincount(s[pred], minlength=len(rec))[s1_c]
        pool = ((rec.source != 1) & (rec.country == c)).sum()
        rows_out.append({"country": c, "S1": len(s1_c), "pred_matches_per_S1": k.mean(), "pred_singleton_rate": (k == 0).mean(),
                         "pool_assigned": np.unique(q[pred & (country[q] == c)]).size / max(pool, 1),
                         "band_pairs": int(np.isin(s[rows], s1_c).sum())})
    log("=== Test checks (train truth: 3.46 matches per S1, 5.6% singletons) ===\n"
        + pd.DataFrame(rows_out).set_index("country").round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["band", "train", "score", "blend", "apply"])
    ap.add_argument("--tag", default="train", choices=["train", "aug", "test"])
    ap.add_argument("--matcher", default=MATCHER)
    ap.add_argument("--workers", type=int, default=C.WORKERS)
    ap.add_argument("--deadline", default="", help="train: optional hard stop (local time HH:MM); none by default")
    ap.add_argument("--experiment", default="m3_gap_ce_fr_u15")
    ap.add_argument("--out-dir", default="output_ce")
    ap.add_argument("--unseen-shift", type=float, default=0.0)
    ap.add_argument("--lookalike-shift", type=float, default=0.0)
    ap.add_argument("--ce-name", default="ce", help="band/train/score: which cross-encoder (model/<name>/)")
    ap.add_argument("--ce-names", default="ce", help="blend/apply: comma list of cross-encoders to blend")
    ap.add_argument("--sample", default="positives_first", choices=["positives_first", "natural"],
                    help="band: training sample (natural = the band's own positive rate)")
    ap.add_argument("--reuse", action="store_true", help="band: reuse ce_gbm.parquet / ce_band.parquet")
    ap.add_argument("--band-lo", type=float, default=BAND_LO)
    ap.add_argument("--band-hi", type=float, default=BAND_HI)
    ap.add_argument("--band-name", default="", help="band/score/blend/apply: name of a non-default band (file suffix)")
    ap.add_argument("--no-train-sample", action="store_true", help="band: do not write a cross-encoder training sample")
    ap.add_argument("--cap", type=int, default=TRAIN_CAP, help="band: training sample size (band + easy)")
    ap.add_argument("--dump-pred", action="store_true", help="blend: write the holdout predictions (q, s)")
    ap.add_argument("--seed", type=int, default=C.SEED, help="band sample / train: random seed")
    ap.add_argument("--swap", action="store_true", help="train/score: candidate text first (same flag for both)")
    a = ap.parse_args()
    {"band": band_main, "train": train_main, "score": score_main, "blend": blend_main, "apply": apply_main}[a.cmd](a)


if __name__ == "__main__":
    main()
