"""Stage 1b (M2): learn normalization tables from training-fold matches only.

  python -m src.v2.learn          # reads runs/v2/train/records_raw.parquet (prepare --raw)

Per country, from matched pairs whose S1 is in the training fold:
- part rewrites: the two addresses differ in exactly one comma part, and the
  query's part is related to the S1 part (abbreviation, initials, consonant
  subsequence, same consonant skeleton, or edit distance 1 for length >= 5);
- word rewrites: the same at word level (exactly one differing word each side);
- native-script dictionary: for non-Latin query names, transliterated tokens
  aligned to S1 name tokens (by position when the counts match, else best
  character similarity).
Rewrites map the query's form to the S1 form. Kept: support >= min and
precision >= min (share of the key's differing occurrences that map to that
value). Writes model/learned_normalization.pkl as {country: kwargs for
normalize_record}; prepare (second pass) applies them to train and test.
"""
from __future__ import annotations

import argparse
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from src.v2 import config as C
from src.v2.normalize import skeleton

REWRITE_SUPPORT, REWRITE_PRECISION = 50, 0.95
DICT_SUPPORT, DICT_PRECISION = 20, 0.90


def _subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(c in it for c in short)


def related(x: str, y: str) -> bool:
    """One form abbreviates or misspells the other."""
    if x == y or not x or not y or any(c.isdigit() for c in x + y):
        return False
    short, long = sorted((x, y), key=len)
    if len(short) >= 2 and long.startswith(short):
        return True
    if len(short) >= 2 and " " in long and short.replace(" ", "") == "".join(w[0] for w in long.split()):
        return True
    if len(short) >= 2 and short[0] == long[0] and " " not in short and _subsequence(short, long.replace(" ", "")):
        return True
    if skeleton(x) and skeleton(x) == skeleton(y):
        return True
    return min(len(x), len(y)) >= 5 and Levenshtein.distance(x, y) <= 1


def select(counts: Counter, totals: Counter, support: int, precision: float) -> dict[str, str]:
    """Best value per key if it passes support and precision."""
    best: dict[str, tuple[str, int]] = {}
    for (k, v), n in counts.items():
        if n > best.get(k, ("", 0))[1]:
            best[k] = (v, n)
    return {k: v for k, (v, n) in best.items() if n >= support and n / totals[k] >= precision}


def learn_rewrites(pairs: pd.DataFrame) -> tuple[dict, dict, dict]:
    part_cnt, part_tot, word_cnt, word_tot = (defaultdict(Counter) for _ in range(4))
    for country, qp, sp, qa, sa in zip(pairs.country, pairs.q_parts, pairs.s_parts, pairs.q_addr, pairs.s_addr):
        A, B = set(filter(None, sp.split(" | "))), set(filter(None, qp.split(" | ")))
        da, db = A - B, B - A
        if len(da) == 1 and len(db) == 1:
            a, b = next(iter(da)), next(iter(db))
            part_tot[country][b] += 1
            if related(b, a):
                part_cnt[country][(b, a)] += 1
        wa, wb = set(sa.split()) - set(qa.split()), set(qa.split()) - set(sa.split())
        if len(wa) == 1 and len(wb) == 1:
            a, b = next(iter(wa)), next(iter(wb))
            word_tot[country][b] += 1
            if related(b, a):
                word_cnt[country][(b, a)] += 1
    parts = {c: select(part_cnt[c], part_tot[c], REWRITE_SUPPORT, REWRITE_PRECISION) for c in part_tot}
    words = {c: select(word_cnt[c], word_tot[c], REWRITE_SUPPORT, REWRITE_PRECISION) for c in word_tot}
    return parts, words, {"part_counts": part_cnt, "word_counts": word_cnt}


def align(q_tokens: list[str], s_tokens: list[str]) -> list[tuple[str, str]]:
    if not q_tokens or not s_tokens:
        return []
    if len(q_tokens) == len(s_tokens):
        return list(zip(q_tokens, s_tokens))
    out = []
    for t in q_tokens:
        best = max(s_tokens, key=lambda w: fuzz.ratio(t, w))
        if fuzz.ratio(t, best) >= 40 or skeleton(t) == skeleton(best):
            out.append((t, best))
    return out


def learn_dictionary(pairs: pd.DataFrame) -> dict:
    cnt, tot = defaultdict(Counter), defaultdict(Counter)
    for country, qn, sn in zip(pairs.country, pairs.q_name, pairs.s_name):
        for t, w in align(qn.split(), sn.split()):
            tot[country][t] += 1
            if t != w:
                cnt[country][(t, w)] += 1
    return {c: select(cnt[c], tot[c], DICT_SUPPORT, DICT_PRECISION) for c in tot}, cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    out = C.tag_dir("train")
    log = C.log_to(f"{out}/learn.log")
    t0 = C.stage_start(log, "learn")
    rec = pd.read_parquet(f"{out}/records_raw.parquet",
                          columns=["rid", "source", "country", "name", "non_latin", "addr", "addr_parts", "fold", "owner"])
    q = rec[(rec.source != 1) & (rec.owner >= 0)]
    s_fold = rec.fold.to_numpy()[q.owner.to_numpy()]
    q = q[s_fold == "train"]
    s = rec.iloc[q.owner.to_numpy()]
    pairs = pd.DataFrame({"country": q.country.to_numpy(), "q_parts": q.addr_parts.to_numpy(), "s_parts": s.addr_parts.to_numpy(),
                          "q_addr": q.addr.to_numpy(), "s_addr": s.addr.to_numpy(), "q_name": q.name.to_numpy(),
                          "s_name": s.name.to_numpy(), "q_non_latin": q.non_latin.to_numpy()})
    log(f"training-fold matched pairs: {len(pairs):,}; non-Latin query names: {pairs.q_non_latin.sum():,}")
    parts, words, _ = learn_rewrites(pairs)
    word_dict, dict_counts = learn_dictionary(pairs[pairs.q_non_latin])
    learned = {}
    for country in sorted(set(parts) | set(words) | set(word_dict)):
        learned[country] = {"part_rewrites": parts.get(country, {}), "rewrites": words.get(country, {}),
                            "word_dict": word_dict.get(country, {})}
        log(f"\n=== {country}: {len(learned[country]['part_rewrites'])} part rewrites, "
            f"{len(learned[country]['rewrites'])} word rewrites, {len(learned[country]['word_dict'])} dictionary entries")
        for title, table in (("part rewrites", parts.get(country, {})), ("word rewrites", words.get(country, {}))):
            log(f"--- {title} (all)")
            for k, v in sorted(table.items()):
                log(f"    {k!r} -> {v!r}")
        top = sorted(word_dict.get(country, {}).items(), key=lambda kv: -dict_counts[country][kv])[:100]
        log("--- dictionary (top 100 by support)")
        for k, v in top:
            log(f"    {k!r} -> {v!r}  ({dict_counts[country][(k, v)]})")
    with open(f"{C.model_dir()}/learned_normalization.pkl", "wb") as f:
        pickle.dump(learned, f)
    C.stage_end(log, "learn", t0)


if __name__ == "__main__":
    main()
