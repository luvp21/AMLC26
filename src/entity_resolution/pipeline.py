"""Orchestrates blocking across name AND address tokens together.

Empirically validated against real training data (notebooks/er_blocking_validation.py,
see docs/ENTITY_RESOLUTION_APPROACH.md for the numbers): name-token-only blocking
recalls only ~73% of true matches at top_k=20, because a meaningful chunk of
Source 2 business names are in Devanagari script — `anyascii` romanizes them
phonetically (e.g. "प्राइम" -> "praim"), which doesn't match the English
spelling in Source 1 ("Prime"). But the *address* field frequently keeps
city/area names in Latin script even when the name doesn't. Blocking on name
tokens AND address tokens together recovers most of that gap: 89.4% recall
at the same top_k=20, 92.0% at top_k=50.
"""
from __future__ import annotations

import multiprocessing as mp

import pandas as pd

from src.entity_resolution.block import build_inverted_index, compute_idf_weights, generate_candidates
from src.entity_resolution.normalize import normalize_address, normalize_name, normalize_record, tokenize

# Set by generate_combined_candidates_parallel BEFORE forking worker processes,
# so children inherit these via copy-on-write instead of having them pickled
# and sent through a pipe per-task (or even once per worker via initargs) —
# the indexes can be large at full dataset scale, and COW inheritance is
# effectively free where explicit passing would not be.
_worker_indexes: list[dict[str, list[str]]] = []
_worker_weights: list[dict[str, float]] | None = []
_worker_top_k: int = 50


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds name_tokens/addr_tokens columns in place (returns the same df).
    Single-threaded — fine up to a few hundred thousand rows. At full
    ~10M-row scale, use add_normalized_columns_parallel instead (single-
    threaded normalize alone would take ~17 minutes for the full dataset)."""
    df["name_tokens"] = df.business_name.map(normalize_name).map(tokenize)
    df["addr_tokens"] = df.business_address.map(normalize_address).map(tokenize)
    return df


def _normalize_row(args: tuple[str, str]) -> tuple[list[str], list[str]]:
    name, addr = args
    return tokenize(normalize_name(name)), tokenize(normalize_address(addr))


def add_normalized_columns_parallel(df: pd.DataFrame, n_workers: int | None = None) -> pd.DataFrame:
    """Same as add_normalized_columns, but parallelized across processes —
    use this at full dataset scale (millions of rows)."""
    n_workers = n_workers or min(32, mp.cpu_count())
    pairs = list(zip(df.business_name, df.business_address))
    with mp.get_context("fork").Pool(n_workers) as pool:
        results = pool.map(_normalize_row, pairs, chunksize=2000)
    df["name_tokens"] = [r[0] for r in results]
    df["addr_tokens"] = [r[1] for r in results]
    return df


_worker_norm_cfg = None


def _normalize_record_row(args: tuple[str, str, str]) -> dict:
    return normalize_record(*args, _worker_norm_cfg)


def add_normalized_columns_cfg(df: pd.DataFrame, cfg, n_workers: int | None = None) -> pd.DataFrame:
    """Phase 2 normalization under NormalizationConfig `cfg` (all flags off ==
    add_normalized_columns_parallel). Adds name_tokens, addr_tokens and, when
    flagged, addr_landmark / postal / postal_all / house_nums / name_core.
    business_name / business_address are left untouched as the raw strings."""
    global _worker_norm_cfg
    _worker_norm_cfg = cfg
    n_workers = n_workers or min(32, mp.cpu_count())
    rows = list(zip(df.business_name, df.business_address, df.country))
    with mp.get_context("fork").Pool(n_workers) as pool:
        results = pool.map(_normalize_record_row, rows, chunksize=2000)
    out = df.copy()
    for col in results[0] if results else ("name_tokens", "addr_tokens"):
        out[col] = [r[col] for r in results]
    return out


def build_source_indexes(
    df: pd.DataFrame, name_max_doc_freq: float = 0.02, addr_max_doc_freq: float = 0.05
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, float], dict[str, float]]:
    """(name_index, address_index, name_weights, address_weights) for one
    source. Address tokens get a looser purge threshold than name tokens —
    city/state names are legitimately common (many businesses share a city)
    but still useful for blocking, unlike generic name words. Weights are
    IDF (rarer token -> higher weight) — see block.compute_idf_weights and
    the module docstring above for why raw shared-token counting isn't
    enough at full dataset scale."""
    ids = df.entity_id.tolist()
    n_docs = len(ids)
    name_index = build_inverted_index(ids, df.name_tokens.tolist(), max_doc_freq=name_max_doc_freq)
    addr_index = build_inverted_index(ids, df.addr_tokens.tolist(), max_doc_freq=addr_max_doc_freq)
    name_weights = compute_idf_weights(name_index, n_docs)
    addr_weights = compute_idf_weights(addr_index, n_docs)
    return name_index, addr_index, name_weights, addr_weights


def flatten_source_indexes(
    source_indexes: tuple[tuple[dict, dict, dict, dict], ...],
) -> tuple[list[dict[str, list[str]]], list[dict[str, float]]]:
    flat_indexes: list[dict[str, list[str]]] = []
    flat_weights: list[dict[str, float]] = []
    for name_idx, addr_idx, name_w, addr_w in source_indexes:
        flat_indexes.extend([name_idx, addr_idx])
        flat_weights.extend([name_w, addr_w])
    return flat_indexes, flat_weights


def generate_combined_candidates(
    s1_df: pd.DataFrame,
    *source_indexes: tuple[dict[str, list[str]], dict[str, list[str]], dict[str, float], dict[str, float]],
    top_k: int = 50,
    use_weights: bool = True,
) -> dict[str, list[str]]:
    """Candidates per Source 1 entity_id, ranked by IDF-weighted combined
    name+address shared-token score (or raw shared-token count if
    `use_weights=False` — for reproducing a specific already-validated run,
    e.g. the submission generated from the 0.8244-scoring model, which was
    trained on features from raw-count-ranked candidates), pooled across
    every given source's (name_index, addr_index, name_weights, addr_weights)
    tuple — pass one per source (e.g. Source 2 and Source 3), as returned by
    build_source_indexes.
    """
    combined_tokens = [n + a for n, a in zip(s1_df.name_tokens, s1_df.addr_tokens)]
    flat_indexes, flat_weights = flatten_source_indexes(source_indexes)
    return generate_candidates(
        s1_df.entity_id.tolist(),
        combined_tokens,
        *flat_indexes,
        weights=flat_weights if use_weights else None,
        top_k=top_k,
    )


def _generate_chunk(args: tuple[list[str], list[list[str]]]) -> dict[str, list[str]]:
    ids_chunk, tokens_chunk = args
    return generate_candidates(
        ids_chunk, tokens_chunk, *_worker_indexes, weights=_worker_weights, top_k=_worker_top_k
    )


def generate_combined_candidates_parallel(
    s1_df: pd.DataFrame,
    *source_indexes: tuple[dict[str, list[str]], dict[str, list[str]], dict[str, float], dict[str, float]],
    top_k: int = 50,
    n_workers: int | None = None,
    use_weights: bool = True,
) -> dict[str, list[str]]:
    """Same as generate_combined_candidates, but splits Source 1 entities into
    chunks processed in parallel — necessary at full dataset scale, where
    per-entity candidate lookup cost grows with index size and a single-
    threaded pass over millions of entities can take many hours (measured,
    not assumed — see docs/ENTITY_RESOLUTION_APPROACH.md). `use_weights=False`
    reproduces the pre-IDF-fix raw-count ranking exactly."""
    global _worker_indexes, _worker_weights, _worker_top_k
    n_workers = n_workers or min(32, mp.cpu_count())
    _worker_indexes, flat_weights = flatten_source_indexes(source_indexes)
    _worker_weights = flat_weights if use_weights else None
    _worker_top_k = top_k

    ids = s1_df.entity_id.tolist()
    combined_tokens = [n + a for n, a in zip(s1_df.name_tokens, s1_df.addr_tokens)]
    chunk_size = max(1, len(ids) // (n_workers * 4))
    chunks = [
        (ids[i : i + chunk_size], combined_tokens[i : i + chunk_size])
        for i in range(0, len(ids), chunk_size)
    ]

    # fork (not spawn): workers must inherit _worker_indexes via copy-on-write,
    # set above just before this call.
    with mp.get_context("fork").Pool(n_workers) as pool:
        results = pool.map(_generate_chunk, chunks)

    combined: dict[str, list[str]] = {}
    for chunk_result in results:
        combined.update(chunk_result)
    return combined
