"""Stage 2: reverse blocking. Every S2/S3 record (query) searches S1 records of
its own country with char 2-4 gram TF-IDF (sparse_dot_topn, top-k per query).

  python -m src.v2.blocking --tag train        # all train queries vs full train S1 index; learns routing
  python -m src.v2.blocking --tag dev          # whole partitions covering ~10% of each country's S1
  python -m src.v2.blocking --tag test         # routing learned on train; unseen countries search everything

Indexes (config.CHANNELS / V2_CHANNELS): n = full cleaned name (top 5; 30 for
empty-address queries), b = name_core + address (top 10), a = address only
(top 10, non-empty addresses), k = consonant skeleton, char 3-4 grams (top 5,
non-Latin queries only). Routing: a query whose partition key has a
learned route searches only S1 records in the routed keys (plus key-less S1);
anything else (no key, unseen key, countries without training data) searches
the whole country. Output OUT_DIR/<tag>/pairs_block.parquet sorted by (q, s):
q, s (int32 record codes), <ch>_score, <ch>_rank per channel (missing: 0 / 99).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from rapidfuzz import fuzz

from src.v2 import config as C
from src.v2.common import channels_of

CHANNELS = tuple(C.CHANNELS)
MISSING_RANK = 99
_VEC: TfidfVectorizer | None = None


def texts(rec: pd.DataFrame, channel: str) -> list[str]:
    if channel == "n":
        return rec.name.tolist()
    if channel == "b":
        return (rec.name_core + " " + rec.addr).str.strip().tolist()
    if channel == "a":
        return rec.addr.tolist()
    if channel == "k":
        return rec.name_skeleton.tolist()
    raise ValueError(f"unknown channel {channel!r}")


def query_topk(q: pd.DataFrame, channel: str) -> list[tuple[np.ndarray, int]]:
    """(query positions, top k) groups for a channel: n uses a larger k for
    empty-address queries, a skips them, k only takes non-Latin queries."""
    empty = (q.addr == "").to_numpy()
    if channel == "n":
        return [(np.flatnonzero(~empty), C.TOPK["n"]), (np.flatnonzero(empty), C.TOPK["n_empty_addr"])]
    if channel == "a":
        return [(np.flatnonzero(~empty), C.TOPK["a"])]
    if channel == "k":
        return [(np.flatnonzero(q.non_latin.to_numpy()), C.TOPK["k"])]
    return [(np.arange(len(q)), C.TOPK[channel])]


def _transform(chunk: list[str]):
    return _VEC.transform(chunk)


def fit_vectorizer(fit_texts: list[str], seed: int, channel: str = "n") -> TfidfVectorizer:
    """The one vocabulary procedure: up to VOCAB_SAMPLE sampled non-empty texts,
    C.TFIDF settings (with the channel's n-gram range)."""
    fit_texts = [t for t in fit_texts if t]
    rng = np.random.default_rng(seed)
    sample = fit_texts if len(fit_texts) <= C.VOCAB_SAMPLE else [
        fit_texts[i] for i in rng.choice(len(fit_texts), C.VOCAB_SAMPLE, replace=False)]
    return TfidfVectorizer(**{**C.TFIDF, **C.TFIDF_OVERRIDES.get(channel, {})}).fit(sample)


def vectorizer_for(country: str, channel: str, saved: dict, fit_texts: list[str]) -> tuple[TfidfVectorizer, str]:
    """Train vocabulary when the country has one (reused on test); otherwise a
    vocabulary fitted on this split's own text with the identical procedure."""
    if channel in saved.get(country, {}):
        return saved[country][channel], "train vocabulary"
    return fit_vectorizer(fit_texts, C.SEED, channel), "fitted on this split's own text"


def vectorize(vec: TfidfVectorizer, all_texts: list[str], workers: int):
    """Transform everything in parallel with a fitted vectorizer."""
    global _VEC
    _VEC = vec
    size = max(1, len(all_texts) // (workers * 4) + 1)
    chunks = [all_texts[i:i + size] for i in range(0, len(all_texts), size)]
    import scipy.sparse as sp
    with mp.get_context("fork").Pool(workers) as pool:
        return sp.vstack(pool.map(_transform, chunks)).tocsr()


def learn_routing(rec: pd.DataFrame) -> dict:
    """{country: {query_key: tuple(S1 keys)}} from training-fold matches: the
    smallest set of S1 keys covering ROUTE_COVERAGE of that query key's matches
    (matches whose owner has no key count as covered: key-less S1 are always searched)."""
    s1 = rec[rec.source == 1]
    s1_key = pd.Series(s1.key.to_numpy(), index=s1.rid.to_numpy())
    s1_fold = pd.Series(s1.fold.to_numpy(), index=s1.rid.to_numpy())
    q = rec[(rec.source != 1) & (rec.owner >= 0) & (rec.key != "")]
    q = q[s1_fold.reindex(q.owner.to_numpy()).to_numpy() == "train"]
    routing: dict = defaultdict(dict)
    counts = Counter(zip(q.country, q.key, s1_key.reindex(q.owner.to_numpy()).to_numpy()))
    per_qkey: dict = defaultdict(Counter)
    for (country, qk, sk), n in counts.items():
        per_qkey[(country, qk)][sk] += n
    for (country, qk), cnt in per_qkey.items():
        total = sum(cnt.values())
        if total < C.ROUTE_MIN_MATCHES:
            continue
        covered, keys = cnt.get("", 0), []
        for sk, n in cnt.most_common():
            if covered / total >= C.ROUTE_COVERAGE:
                break
            if sk:
                keys.append(sk)
                covered += n
        routing[country][qk] = tuple(sorted(keys))
    return dict(routing)


def _place_parts(parts: str) -> list[str]:
    return [p for p in parts.split(" | ") if p and not any(c.isdigit() for c in p)]


def unsupervised_routing(s1: pd.DataFrame, q: pd.DataFrame) -> tuple[np.ndarray, list]:
    """Label-free routing for a country without learned routes (no training data).

    S1 keys: each S1's most frequent digit-free address part among parts that
    occur >= KEY_MIN_COUNT times in the country's S1 (a region, as with states).
    Place table: a part (city, department, region) maps to an S1 key when, over
    the records where it occurs together with a key, one key holds >= UNSUP_PURITY
    of >= UNSUP_MIN_SUPPORT occurrences. Learned from S1 first, then extended with
    parts of any record (S1 or query) whose known parts resolve to one key (e.g. a
    department that only queries write). A query routes to the keys of its parts
    (at most UNSUP_MAX_KEYS); anything else searches the whole country. If the
    S1 self-check miss rate exceeds UNSUP_MAX_SELF_MISS (messy keys, e.g. cities
    competing with states), nothing is routed.
    Returns (S1 keys, per-query route tuple or None, self-check miss rate)."""
    s1_parts = [_place_parts(p) for p in s1.addr_parts]
    q_parts = [_place_parts(p) for p in q.addr_parts]
    freq = Counter(p for ps in s1_parts for p in ps)
    vocab = {p: n for p, n in freq.items() if n >= C.KEY_MIN_COUNT}
    s1_key = np.array([max((p for p in ps if p in vocab), key=vocab.get, default="") for ps in s1_parts], dtype=object)

    def table_from(parts_list, keys) -> dict:
        co: dict = defaultdict(Counter)
        for ps, k in zip(parts_list, keys):
            if k:
                for p in set(ps):
                    co[p][k] += 1
        out = {}
        for p, cnt in co.items():
            total = sum(cnt.values())
            k, n = cnt.most_common(1)[0]
            if total >= C.UNSUP_MIN_SUPPORT and n / total >= C.UNSUP_PURITY:
                out[p] = k
        return out

    def resolve(ps, table):
        keys = {table[p] for p in ps if p in table}
        return tuple(sorted(keys)) if 0 < len(keys) <= C.UNSUP_MAX_KEYS else None

    table = table_from(s1_parts, s1_key)
    all_parts = s1_parts + q_parts
    first = [resolve(ps, table) for ps in all_parts]
    single = [r[0] if r is not None and len(r) == 1 else "" for r in first]
    table = {**table_from(all_parts, single), **table}  # S1-learned entries win
    # label-free check: an S1's own places, routed like a query, must lead back to its key
    # (a true match writes the same places); the miss rate estimates the recall routing costs
    own = [resolve(ps, table) for ps in s1_parts]
    self_miss = float(np.mean([r is not None and bool(k) and k not in r for r, k in zip(own, s1_key)]))
    if self_miss > C.UNSUP_MAX_SELF_MISS:
        return s1_key, [None] * len(q_parts), self_miss
    return s1_key, [resolve(ps, table) for ps in q_parts], self_miss


def _topn_arrays(m, q_rids, idx_rids):
    m = m.tocsr()
    row = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
    order = np.lexsort((-m.data, row))
    row, col, data = row[order], m.indices[order], m.data[order]
    rank = np.arange(len(data)) - m.indptr[row]
    return q_rids[row], idx_rids[col], data.astype(np.float32), rank.astype(np.int16)


def search_country(s1: pd.DataFrame, q: pd.DataFrame, route: dict, workers: int, log, country: str,
                   saved_vecs: dict, fitted_vecs: dict, q_route: list | None = None) -> pd.DataFrame:
    """q_route (per-query S1-key tuple or None) overrides routing by `route` and q.key."""
    fit_all = {ch: texts(pd.concat([s1, q]), ch) for ch in CHANNELS}
    mats = {}
    for ch in CHANNELS:
        t0 = time.time()
        vec, source = vectorizer_for(country, ch, saved_vecs, fit_all[ch])
        fitted_vecs.setdefault(country, {})[ch] = vec
        m = vectorize(vec, fit_all[ch], workers)
        mats[ch] = (m[: len(s1)], m[len(s1):])
        log(f"    vectorized {ch} ({source}): {m.shape[1]:,} n-grams, {time.time() - t0:.0f}s")
    s1_rid, q_rid = s1.rid.to_numpy(), q.rid.to_numpy()
    s1_key = s1.key.to_numpy()
    keyless = s1_key == ""
    if q_route is None:
        q_route = [route.get(k) if k else None for k in q.key]
    groups: dict = defaultdict(list)
    for i, r in enumerate(q_route):
        groups[r].append(i)
    q_groups = {ch: query_topk(q, ch) for ch in CHANNELS}
    in_group = {ch: [np.isin(np.arange(len(q)), pos) for pos, _ in q_groups[ch]] for ch in CHANNELS}
    out = {ch: [] for ch in CHANNELS}
    index_sizes = np.zeros(len(q), dtype=np.int64)
    t0 = time.time()
    n_done, next_report = 0, 0.1
    for r, qpos in groups.items():
        n_done += len(qpos)
        if n_done / len(q) >= next_report:
            log(f"    search progress {n_done / len(q):.0%} of queries, {time.time() - t0:.0f}s")
            next_report += 0.1
        qpos = np.asarray(qpos)
        idx = np.flatnonzero(keyless | np.isin(s1_key, r)) if r is not None else np.arange(len(s1))
        index_sizes[qpos] = len(idx)
        if len(idx) == 0:
            continue
        for ch in CHANNELS:
            index_t = mats[ch][0][idx].T.tocsr()
            for start in range(0, len(qpos), C.QUERY_CHUNK):
                chunk = qpos[start:start + C.QUERY_CHUNK]
                subsets = [(chunk[mask[chunk]], k) for mask, (_, k) in zip(in_group[ch], q_groups[ch])]
                for sub, k in subsets:
                    if len(sub) == 0:
                        continue
                    m = sp_matmul_topn(mats[ch][1][sub], index_t, top_n=k, threshold=C.SCORE_FLOOR,
                                       sort=True, n_threads=workers)
                    out[ch].append(_topn_arrays(m, q_rid[sub], s1_rid[idx]))
    log(f"    searched {len(groups):,} route groups in {time.time() - t0:.0f}s; "
        f"avg index size per query {index_sizes.mean():,.0f} (whole country {len(s1):,})")
    return union(out)


def union(out: dict) -> pd.DataFrame:
    """Union of the channels in `out` per (q, s), each channel's score and rank
    (score 0 / rank MISSING_RANK where it did not return the pair)."""
    channels = list(out)
    parts = {}
    for ch in channels:
        arrs = out[ch]
        if arrs:
            qq, ss, sc, rk = (np.concatenate(x) for x in zip(*arrs))
        else:
            qq = ss = np.zeros(0, np.int32)
            sc, rk = np.zeros(0, np.float32), np.zeros(0, np.int16)
        parts[ch] = ((qq.astype(np.int64) << 32) | ss.astype(np.int64), sc, rk)
    keys = np.unique(np.concatenate([parts[ch][0] for ch in channels]))
    df = pd.DataFrame({"q": (keys >> 32).astype(np.int32), "s": (keys & 0xFFFFFFFF).astype(np.int32)})
    for ch in channels:
        k, sc, rk = parts[ch]
        pos = np.searchsorted(keys, k)
        score = np.zeros(len(keys), np.float32)
        rank = np.full(len(keys), MISSING_RANK, np.int16)
        score[pos], rank[pos] = sc, rk
        df[f"{ch}_score"], df[f"{ch}_rank"] = score, rank
    return df


def dev_subset(s1: pd.DataFrame, q: pd.DataFrame, route: dict, frac: float = 0.10):
    """Whole partitions (S1 keys) covering ~frac of the country's S1, with every
    S1 in them and every query routed to them; unrouted queries kept at rate frac."""
    sizes = s1[s1.key != ""].key.value_counts()
    order = sorted(sizes.index, key=lambda k: C.md5_mod([k], 1 << 30, "devkey|")[0])
    chosen, total = set(), 0
    for k in order:
        if total >= frac * len(s1):
            break
        chosen.add(k)
        total += sizes[k]
    s1_dev = s1[s1.key.isin(chosen)]
    routes = [route.get(k) if k else None for k in q.key]
    h = C.md5_mod(q.entity_id.tolist(), 100, "dev|")
    keep = np.array([(r is None and hh < frac * 100) or (r is not None and bool(chosen.intersection(r)))
                     for r, hh in zip(routes, h)])
    return s1_dev, q[keep], chosen


def pair_category(q_core, s_core, q_addr, q_non_latin, alias, crowd) -> str:
    """Heuristic bucket of a true pair (first match wins)."""
    if not q_addr:
        return "empty_address"
    if q_non_latin:
        return "non_latin"
    if alias:
        return "trade_name_alias"
    sim = fuzz.token_set_ratio(q_core, s_core)
    if sim < 40:
        return "scrambled_name"
    if q_core == s_core and crowd >= 5:
        return "same_name_crowd"
    if q_core != s_core and sim >= 80:
        return "typo"
    return "other"


def recall_report(pairs: pd.DataFrame, rec: pd.DataFrame, s1_pool: pd.DataFrame, country: str, route: dict, log) -> dict:
    """Holdout entities of the searched S1 pool: pair recall, entity all-match
    recall, candidates per S1, unique recall per channel, recall per category,
    and true pairs lost to routing (owner outside the query's routed keys)."""
    s1 = s1_pool[s1_pool.fold == "holdout"]
    hold = set(s1.rid)
    q = rec[(rec.source != 1) & rec.owner.isin(hold)]
    own = rec.iloc[q.owner.to_numpy()]
    true_keys = (q.rid.to_numpy().astype(np.int64) << 32) | q.owner.to_numpy().astype(np.int64)
    got_keys = (pairs.q.to_numpy().astype(np.int64) << 32) | pairs.s.to_numpy().astype(np.int64)
    pos = np.searchsorted(got_keys, true_keys).clip(0, max(len(got_keys) - 1, 0))
    hit = (got_keys[pos] == true_keys) if len(got_keys) else np.zeros(len(true_keys), bool)
    chan_hit = {ch: hit & (pairs[f"{ch}_rank"].to_numpy()[pos] < MISSING_RANK) for ch in channels_of(pairs)}
    per_s1 = pd.Series(hit).groupby(q.owner.to_numpy()).agg(["sum", "size"])
    cand = pairs[pairs.s.isin(hold)].groupby("s").size().reindex(s1.rid, fill_value=0)
    routes = [route.get(k) if k else None for k in q.key]
    routed_out = np.array([r is not None and bool(sk) and sk not in r for r, sk in zip(routes, own.key)])
    crowd = s1_pool.name_core.value_counts()
    cats = np.array([pair_category(qc, sc, qa, nl, bool(qa_ or sa_), crowd.get(sc, 0)) for qc, sc, qa, nl, qa_, sa_ in zip(
        q.name_core, own.name_core, q.addr, q.non_latin, q.name_alias, own.name_alias)])
    stats = {"country": country, "true_pairs": len(q), "pair_recall": hit.mean() if len(q) else np.nan,
             "entity_all_recall": (per_s1["sum"] == per_s1["size"]).mean() if len(per_s1) else np.nan,
             "cand_per_s1": cand.mean(), "cand_p95": cand.quantile(0.95), "routing_lost": routed_out.mean()}
    for ch, h in chan_hit.items():
        others = np.any([v for c, v in chan_hit.items() if c != ch], axis=0) if len(chan_hit) > 1 else np.zeros(len(h), bool)
        stats[f"only_{ch}"] = (h & ~others).mean()
    log("    " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in stats.items()))
    table = pd.DataFrame({"category": cats, "hit": hit}).groupby("category").hit.agg(["size", "mean"])
    table.columns = ["true_pairs", "recall"]
    log("    recall by category:\n" + "\n".join("      " + line for line in table.round(4).to_string().splitlines()))
    for cat, row in table.iterrows():
        stats[f"recall_{cat}"] = row.recall
    return stats


def unsup_eval(a) -> None:
    """How label-free routing would do on countries with labels: share of true
    pairs whose owner falls outside the query's route, and index size per query."""
    rec = pd.read_parquet(f"{C.tag_dir('train')}/records.parquet",
                          columns=["rid", "source", "country", "addr_parts", "owner"])
    log = C.log_to(f"{C.tag_dir('train')}/unsup_routing.log")
    countries = [c for c in a.countries.split(",") if c] or sorted(rec.country.unique())
    for country in countries:
        s1 = rec[(rec.source == 1) & (rec.country == country)]
        q = rec[(rec.source != 1) & (rec.country == country)]
        s1_key, q_route, self_miss = unsupervised_routing(s1, q)
        key_of = pd.Series(s1_key, index=s1.rid.to_numpy())
        size_of = pd.Series(s1_key).value_counts()
        keyless = int(size_of.get("", 0))
        owned = q.owner.to_numpy() >= 0
        own_key = key_of.reindex(q.owner.to_numpy()).fillna("").to_numpy()
        lost = np.array([r is not None and bool(k) and k not in r for r, k in zip(q_route, own_key)])
        index = np.array([len(s1) if r is None else keyless + sum(size_of.get(k, 0) for k in r) for r in q_route])
        log(f"{country}: {len(set(s1_key) - {''}):,} S1 keys, S1 key coverage {(s1_key != '').mean():.3f}, "
            f"self-check miss {self_miss:.4f}, "
            f"queries routed {np.mean([r is not None for r in q_route]):.3f}, "
            f"true pairs lost to routing {lost[owned].mean():.4f} ({lost[owned].sum():,} of {owned.sum():,}), "
            f"avg index {index.mean():,.0f} of {len(s1):,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--workers", type=int, default=C.WORKERS)
    ap.add_argument("--countries", default="", help="comma list to restrict (e.g. to time one country)")
    ap.add_argument("--unsup-routing", action="store_true",
                    help="countries without learned routes use label-free routing (unsupervised_routing)")
    ap.add_argument("--unsup-eval", action="store_true",
                    help="train only: score label-free routing on the training countries against labels, no search")
    a = ap.parse_args()
    if a.unsup_eval:
        unsup_eval(a)
        return
    split = "test" if a.tag == "test" else "train"
    out = C.tag_dir(a.tag)
    log = C.log_to(f"{out}/blocking.log")
    t_stage = C.stage_start(log, f"blocking {a.tag}")
    rec = pd.read_parquet(f"{C.tag_dir(split)}/records.parquet")
    vec_path = f"{C.model_dir()}/tfidf_vocab.pkl"
    saved_vecs, fitted_vecs = {}, {}
    if a.tag == "test":
        with open(vec_path, "rb") as f:
            saved_vecs = pickle.load(f)
    routing_path = f"{C.model_dir()}/routing.pkl"
    if split == "train":
        routing = learn_routing(rec)
        with open(routing_path, "wb") as f:
            pickle.dump(routing, f)
        for country, r in routing.items():
            log(f"routing {country}: {len(r):,} query keys routed; mean S1 keys per route "
                f"{np.mean([len(v) for v in r.values()]):.2f}")
    else:
        with open(routing_path, "rb") as f:
            routing = pickle.load(f)

    countries = [c for c in a.countries.split(",") if c] or sorted(rec.country.unique())
    frames, stats = [], []
    for country in countries:
        t0 = time.time()
        s1 = rec[(rec.source == 1) & (rec.country == country)]
        q = rec[(rec.source != 1) & (rec.country == country)]
        route = routing.get(country, {})
        if a.tag == "dev":
            s1, q, chosen = dev_subset(s1, q, route)
            log(f"{country} dev partitions: {sorted(chosen)}")
        log(f"{country}: {len(s1):,} S1, {len(q):,} queries, routed keys {len(route):,}")
        q_route = None
        if not route and a.unsup_routing:
            s1_key, q_route, self_miss = unsupervised_routing(s1, q)
            s1 = s1.assign(key=s1_key)
            log(f"  unsupervised routing: {len(set(s1_key) - {''}):,} S1 keys (S1 key coverage {(s1_key != '').mean():.3f}), "
                f"S1 self-check miss {self_miss:.4f}, queries routed {np.mean([r is not None for r in q_route]):.3f}")
        pairs = search_country(s1, q, route, a.workers, log, country, saved_vecs, fitted_vecs, q_route)
        log(f"  {country}: {len(pairs):,} pairs ({len(pairs) / max(len(s1), 1):.1f} per S1) in {time.time() - t0:.0f}s")
        if split == "train":
            stats.append(recall_report(pairs, rec, s1, country, route, log))
        frames.append(pairs)
    pairs = pd.concat(frames, ignore_index=True).sort_values(["q", "s"], ignore_index=True)
    pairs.to_parquet(f"{out}/pairs_block.parquet", index=False)
    if stats:
        pd.DataFrame(stats).to_csv(f"{out}/blocking_recall.csv", index=False)
    if a.tag == "train" and not a.countries:
        with open(vec_path, "wb") as f:
            pickle.dump(fitted_vecs, f)
        log(f"saved train TF-IDF vocabularies for {sorted(fitted_vecs)} -> {vec_path}")
    log(f"wrote {out}/pairs_block.parquet ({len(pairs):,} pairs)")
    C.stage_end(log, f"blocking {a.tag}", t_stage)


if __name__ == "__main__":
    main()
