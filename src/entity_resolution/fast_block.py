"""Vectorized (sparse-matrix) blocking — a much faster replacement for the
per-entity Python dict/Counter loop in block.py.

Measured on the full-scale run: candidate generation took ~4 hours even
parallelized across 32 workers, and dominates the pipeline's total runtime
(everything else combined — normalize, index build, features, training —
is under 35 minutes). Root cause: for each Source 1 entity, the old
approach loops over its tokens in pure Python and, per token, iterates
that token's entire posting list (up to max_postings=20,000 entries) doing
dict increments. That's inherently slow at Python's interpreter speed, no
matter how many processes it's split across.

This does the same computation — IDF-weighted shared-token score between
every Source 1 entity and every candidate — as one (or a few, batched)
sparse matrix multiplications via scipy/scikit-learn, which run in
optimized C/BLAS code instead of a Python loop. This is the standard,
established way to do TF-IDF-weighted retrieval at scale; the original
block.py approach was correct but not the right tool for this data volume.

Preserves the existing design choices from block.py/pipeline.py exactly.
name and addr get independently-fit TF-IDF spaces per source (so their
different max_df purge thresholds still apply as designed), and Source
2/3 stay independently indexed (matching build_source_indexes's existing
per-source behavior) — this is a speed rewrite, not a behavior change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


def _identity_analyzer(tokens: list[str]) -> list[str]:
    return tokens


def build_tfidf_matrix(
    token_lists: list[list[str]],
    max_doc_freq: float = 0.02,
    max_postings: int = 20_000,
) -> tuple[TfidfVectorizer, sp.csr_matrix]:
    """Fit a TF-IDF space on already-tokenized documents (each doc is a
    list of tokens, not a raw string — bypasses sklearn's own tokenizer to
    reuse normalize.py's tokenization exactly). `max_df` mirrors
    build_inverted_index's purge logic: the smaller of the ratio and the
    absolute cap, since 2% of 5M rows is still a 100k-entry list that's
    too generic to be useful (see block.py's docstring)."""
    n_docs = len(token_lists)
    max_df_count = max(1, int(min(max_doc_freq * n_docs, max_postings)))
    vectorizer = TfidfVectorizer(
        analyzer=_identity_analyzer,
        max_df=max_df_count,
        min_df=1,
        binary=True,  # presence, not count — a token rarely repeats within one short name/address
        use_idf=True,
        norm=None,  # keep raw idf-weighted-sum scores, not cosine-normalized — matches block.py's additive scoring
    )
    try:
        matrix = vectorizer.fit_transform(token_lists)
    except ValueError:
        # Every document was empty (e.g. a source with no address data at all) —
        # a real case seen in this dataset (some Source 3 rows have blank
        # addresses), not just a test-fixture edge case. Zero-column matrix:
        # every query scores 0 against this field, which is the correct
        # behavior (nothing to match on).
        matrix = sp.csr_matrix((n_docs, 0))
    return vectorizer, matrix


def top_k_indices_batched(
    query_matrix: sp.csr_matrix, candidate_matrix: sp.csr_matrix, top_k: int, batch_size: int = 5000
) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each row in query_matrix, the (indices, scores) of its top_k
    highest-scoring rows in candidate_matrix, via sparse matrix
    multiplication. Batched so the full (n_queries x n_candidates) score
    matrix is never materialized at once — only one batch's worth."""
    candidate_matrix_t = candidate_matrix.T.tocsr()
    n_queries = query_matrix.shape[0]
    results: list[tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, n_queries, batch_size):
        batch = (query_matrix[start : start + batch_size] @ candidate_matrix_t).tocsr()
        for row_idx in range(batch.shape[0]):
            row = batch.getrow(row_idx)
            if row.nnz == 0:
                results.append((np.array([], dtype=np.int64), np.array([], dtype=np.float64)))
                continue
            if row.nnz <= top_k:
                order = np.argsort(-row.data)
            else:
                top = np.argpartition(-row.data, top_k)[:top_k]
                order = top[np.argsort(-row.data[top])]
            results.append((row.indices[order], row.data[order]))
    return results


def generate_candidates_fast(
    s1_df: pd.DataFrame,
    source_dfs: list[pd.DataFrame],
    top_k: int = 30,
    name_max_doc_freq: float = 0.02,
    addr_max_doc_freq: float = 0.05,
    batch_size: int = 5000,
) -> dict[str, list[str]]:
    """Drop-in faster replacement for pipeline.generate_combined_candidates(_parallel).
    `source_dfs` is a list of per-source DataFrames (e.g. [s2_df, s3_df]),
    each already normalized (name_tokens/addr_tokens columns present, see
    pipeline.add_normalized_columns_parallel).
    """
    s1_ids = s1_df.entity_id.tolist()
    s1_name_tokens = s1_df.name_tokens.tolist()
    s1_addr_tokens = s1_df.addr_tokens.tolist()

    # source_idx -> (candidate_ids, best_score_per_s1) accumulated across sources
    pooled_scores: list[dict[str, float]] = [dict() for _ in s1_ids]

    def safe_transform(vectorizer: TfidfVectorizer, matrix: sp.csr_matrix, token_lists: list[list[str]]):
        # matrix.shape[1] == 0 means build_tfidf_matrix hit an empty vocabulary
        # (e.g. a source with no address data at all — a real case in this
        # dataset) and the vectorizer was never actually fitted; calling
        # .transform() on it would raise. A zero-column query matrix is the
        # correct behavior: nothing to match on, contributes no score.
        if matrix.shape[1] == 0:
            return sp.csr_matrix((len(token_lists), 0))
        return vectorizer.transform(token_lists)

    for source_df in source_dfs:
        cand_ids = source_df.entity_id.to_numpy()
        name_vectorizer, name_matrix = build_tfidf_matrix(
            source_df.name_tokens.tolist(), max_doc_freq=name_max_doc_freq
        )
        addr_vectorizer, addr_matrix = build_tfidf_matrix(
            source_df.addr_tokens.tolist(), max_doc_freq=addr_max_doc_freq
        )

        query_name_matrix = safe_transform(name_vectorizer, name_matrix, s1_name_tokens)
        query_addr_matrix = safe_transform(addr_vectorizer, addr_matrix, s1_addr_tokens)

        # Wider-than-final top_k per source before pooling across sources, so a
        # source that's individually weaker doesn't get starved before pooling.
        per_source_k = min(top_k * 2, name_matrix.shape[0])
        name_hits = top_k_indices_batched(query_name_matrix, name_matrix, per_source_k, batch_size)
        addr_hits = top_k_indices_batched(query_addr_matrix, addr_matrix, per_source_k, batch_size)

        for i in range(len(s1_ids)):
            for indices, scores in (name_hits[i], addr_hits[i]):
                for idx, score in zip(indices, scores):
                    cid = cand_ids[idx]
                    pooled_scores[i][cid] = pooled_scores[i].get(cid, 0.0) + float(score)

    candidates: dict[str, list[str]] = {}
    for s1_id, scores in zip(s1_ids, pooled_scores):
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        candidates[s1_id] = [cid for cid, _ in ranked[:top_k]]
    return candidates
