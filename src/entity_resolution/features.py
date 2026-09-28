"""Pairwise similarity features for a (Source 1 record, candidate) pair —
the input to the matching classifier. All string-based, script-agnostic (no
per-country branching), computed on the already-normalized name/address text
and token lists from `normalize.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

FEATURE_NAMES = [
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaccard",
    "addr_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_jaccard",
    "country_match",
    "name_len_diff",
    "addr_len_diff",
    "shared_name_tokens",
    "shared_addr_tokens",
]


def _jaccard(a: list[str], b: list[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def compute_pair_features(
    s1_name: str, s1_addr: str, s1_name_tokens: list[str], s1_addr_tokens: list[str], s1_country: str,
    c_name: str, c_addr: str, c_name_tokens: list[str], c_addr_tokens: list[str], c_country: str,
) -> list[float]:
    return [
        fuzz.ratio(s1_name, c_name) / 100.0,
        fuzz.token_sort_ratio(s1_name, c_name) / 100.0,
        fuzz.token_set_ratio(s1_name, c_name) / 100.0,
        _jaccard(s1_name_tokens, c_name_tokens),
        fuzz.ratio(s1_addr, c_addr) / 100.0,
        fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0,
        fuzz.token_set_ratio(s1_addr, c_addr) / 100.0,
        _jaccard(s1_addr_tokens, c_addr_tokens),
        1.0 if s1_country == c_country else 0.0,
        float(abs(len(s1_name) - len(c_name))),
        float(abs(len(s1_addr) - len(c_addr))),
        float(len(set(s1_name_tokens) & set(c_name_tokens))),
        float(len(set(s1_addr_tokens) & set(c_addr_tokens))),
    ]


def build_feature_matrix(pairs_df: pd.DataFrame) -> np.ndarray:
    """pairs_df must have columns: s1_name, s1_addr, s1_name_tokens,
    s1_addr_tokens, s1_country, c_name, c_addr, c_name_tokens, c_addr_tokens,
    c_country (one row per candidate pair). Returns a float32 matrix in
    FEATURE_NAMES order."""
    rows = [
        compute_pair_features(
            r.s1_name, r.s1_addr, r.s1_name_tokens, r.s1_addr_tokens, r.s1_country,
            r.c_name, r.c_addr, r.c_name_tokens, r.c_addr_tokens, r.c_country,
        )
        for r in pairs_df.itertuples()
    ]
    return np.array(rows, dtype=np.float32)


def _scores(left: list[str], right: list[str], scorer) -> np.ndarray:
    # float64 then /100, matching the per-row path's Python-float arithmetic,
    # so the final float32 cast is bit-identical to build_feature_matrix.
    return process.cpdist(left, right, scorer=scorer, workers=1, dtype=np.float64) / 100.0


def _overlap(left: list[list[str]], right: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    # Plain Python sets: profiled 3.6x faster than a sparse-incidence version,
    # because sklearn's vocabulary building is itself a Python-level loop.
    jac = np.fromiter((_jaccard(a, b) for a, b in zip(left, right)), dtype=np.float64, count=len(left))
    shared = np.fromiter((len(set(a) & set(b)) for a, b in zip(left, right)), dtype=np.float64, count=len(left))
    return jac, shared


def _features_for_range(bounds: tuple[int, int]) -> np.ndarray:
    chunk = _FEATURE_PAIRS.iloc[bounds[0] : bounds[1]]
    s1_name, c_name = chunk.s1_name.tolist(), chunk.c_name.tolist()
    s1_addr, c_addr = chunk.s1_addr.tolist(), chunk.c_addr.tolist()
    name_jac, name_shared = _overlap(chunk.s1_name_tokens.tolist(), chunk.c_name_tokens.tolist())
    addr_jac, addr_shared = _overlap(chunk.s1_addr_tokens.tolist(), chunk.c_addr_tokens.tolist())
    cols = [
        _scores(s1_name, c_name, fuzz.ratio),
        _scores(s1_name, c_name, fuzz.token_sort_ratio),
        _scores(s1_name, c_name, fuzz.token_set_ratio),
        name_jac,
        _scores(s1_addr, c_addr, fuzz.ratio),
        _scores(s1_addr, c_addr, fuzz.token_sort_ratio),
        _scores(s1_addr, c_addr, fuzz.token_set_ratio),
        addr_jac,
        (chunk.s1_country.to_numpy() == chunk.c_country.to_numpy()).astype(np.float64),
        np.abs(chunk.s1_name.str.len().to_numpy() - chunk.c_name.str.len().to_numpy()).astype(np.float64),
        np.abs(chunk.s1_addr.str.len().to_numpy() - chunk.c_addr.str.len().to_numpy()).astype(np.float64),
        name_shared,
        addr_shared,
    ]
    return np.column_stack(cols).astype(np.float32)


# Set before forking so workers inherit the pairs table copy-on-write instead
# of receiving pickled chunks (same pattern as pipeline.py's parallel blocking).
_FEATURE_PAIRS: pd.DataFrame | None = None


def build_feature_matrix_fast(
    pairs_df: pd.DataFrame, n_workers: int = 1, chunk_size: int = 500_000
) -> np.ndarray:
    """Bit-identical to build_feature_matrix (same FEATURE_NAMES order and
    values), but rapidfuzz scores run through process.cpdist (C++) and the
    pair table is split across worker processes. The per-row version measured
    ~20µs/pair — ~17 minutes single-threaded for the 52M test pairs."""
    global _FEATURE_PAIRS
    if len(pairs_df) == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    _FEATURE_PAIRS = pairs_df
    bounds = [(s, min(s + chunk_size, len(pairs_df))) for s in range(0, len(pairs_df), chunk_size)]
    if n_workers <= 1 or len(bounds) == 1:
        parts = [_features_for_range(b) for b in bounds]
    else:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(n_workers) as pool:
            parts = pool.map(_features_for_range, bounds)
    _FEATURE_PAIRS = None
    return np.vstack(parts)


# Embedding features are optional and separate from FEATURE_NAMES/build_feature_matrix
# above: they need a downloaded model and (ideally) a GPU, whereas the rapidfuzz/Jaccard
# features always work with no extra setup. See src/entity_resolution/embeddings.py for
# why this targets classifier precision on cross-script (Devanagari) name pairs.
EMBEDDING_FEATURE_NAMES = ["name_embedding_cosine", "addr_embedding_cosine"]


def build_full_feature_matrix(
    pairs_df: pd.DataFrame, use_embeddings: bool = False, embedding_model: str | None = None
) -> tuple[np.ndarray, list[str]]:
    """build_feature_matrix's output, optionally with embedding-similarity
    columns appended. Returns (X, feature_names) so callers always know
    which columns are which regardless of which features were included —
    a saved model's feature_importances_ is only meaningful paired with the
    exact feature_names it was trained on."""
    X = build_feature_matrix(pairs_df)
    names = list(FEATURE_NAMES)
    if not use_embeddings:
        return X, names

    from src.entity_resolution.embeddings import DEFAULT_MODEL, build_embedding_features

    X_emb = build_embedding_features(pairs_df, model_name=embedding_model or DEFAULT_MODEL)
    return np.hstack([X, X_emb]), names + EMBEDDING_FEATURE_NAMES
