"""Candidate generation (blocking) for entity resolution.

Standard "token blocking" (Papadakis et al., ACM Computing Surveys 2020):
index every source by the tokens in its normalized business_name, then for a
Source 1 record, candidates are every Source 2/3 record sharing at least one
token. High-document-frequency tokens (generic words like "private",
"limited") are purged before indexing — otherwise they create enormous,
useless blocks. Candidates are then ranked and pruned to the top-K per
Source 1 entity, since the *full* index of ~10M Source 2+3 records makes
exhaustive comparison infeasible.

This determines the recall ceiling for the whole pipeline — see
docs/PROBLEM_STATEMENT.md's note on why blocking quality matters most.

Ranking is IDF-weighted, not raw shared-token count — diagnosed empirically
against the full-scale run (docs/ENTITY_RESOLUTION_APPROACH.md): 97% of
missed true matches were *present* in the unrestricted candidate pool but
buried past the top_k=30 cutoff, because pools of 100k-180k candidates
sharing at least one token get dominated by ties on generic connector words
("no", "city", "delhi") that count exactly the same as distinctive words
("vision", "enterprises"). Weighting each shared token by how rare it is in
the corpus fixes this directly — a shared "vision" now outweighs a shared
"no" by orders of magnitude, which is exactly the discriminating signal raw
counting couldn't express.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict


def build_inverted_index(
    ids: list[str],
    token_lists: list[list[str]],
    max_doc_freq: float = 0.02,
    max_postings: int = 20_000,
) -> dict[str, list[str]]:
    """token -> list of entity_ids whose normalized name contains that token.
    Tokens appearing in more than `max_doc_freq` fraction of records, OR in
    more than `max_postings` records outright, are dropped (block purging) —
    they're too generic to be useful for blocking (e.g. "private", "limited",
    common city names) and would otherwise produce blocks too large to rank
    efficiently. The absolute cap matters as much as the ratio at large
    corpus sizes: 2% of 5M rows is still a 100k-entry posting list, which
    dominates per-entity lookup cost regardless of how rare the token
    "feels" — confirmed as a real bottleneck when scaling from a 300k-row
    validation sample to the full ~10M-row dataset (see
    docs/ENTITY_RESOLUTION_APPROACH.md).
    """
    doc_freq: Counter[str] = Counter()
    for tokens in token_lists:
        doc_freq.update(set(tokens))

    n_docs = len(ids)
    max_count = min(max_doc_freq * n_docs, max_postings)
    purged = {tok for tok, count in doc_freq.items() if count > max_count}

    index: dict[str, list[str]] = defaultdict(list)
    for entity_id, tokens in zip(ids, token_lists):
        for tok in set(tokens):
            if tok not in purged:
                index[tok].append(entity_id)
    return dict(index)


def compute_idf_weights(index: dict[str, list[str]], n_docs: int) -> dict[str, float]:
    """Inverse-document-frequency weight per token already in an inverted
    index — `len(postings)` IS that token's document frequency, so this
    needs no separate frequency tracking. Rarer tokens (more discriminating)
    get higher weight; +1 smoothing keeps a token appearing in every
    remaining document from producing a zero/negative weight."""
    return {tok: math.log(n_docs / len(postings) + 1) for tok, postings in index.items()}


def candidates_for_record(
    tokens: list[str],
    *indexes: dict[str, list[str]],
    weights: list[dict[str, float]] | None = None,
    top_k: int = 20,
    min_shared_tokens: int = 1,
) -> list[str]:
    """Rank candidates (pooled across one or more indexes, e.g. Source 2 and
    Source 3) by IDF-weighted shared-token score with this record's tokens
    (falls back to unweighted/count-based ranking if `weights` is omitted),
    keep top_k. `min_shared_tokens` still filters on raw count, not the
    weighted score — it's a "did they share anything at all" floor, not a
    quality threshold.
    """
    shared_scores: dict[str, float] = defaultdict(float)
    shared_counts: Counter[str] = Counter()
    query_tokens = set(tokens)
    for i, index in enumerate(indexes):
        index_weights = weights[i] if weights is not None else None
        for tok in query_tokens:
            weight = index_weights.get(tok, 1.0) if index_weights is not None else 1.0
            for candidate_id in index.get(tok, ()):
                shared_scores[candidate_id] += weight
                shared_counts[candidate_id] += 1

    ranked = sorted(
        (cid for cid, count in shared_counts.items() if count >= min_shared_tokens),
        key=lambda cid: shared_scores[cid],
        reverse=True,
    )
    return ranked[:top_k]


def generate_candidates(
    s1_ids: list[str],
    s1_token_lists: list[list[str]],
    *indexes: dict[str, list[str]],
    weights: list[dict[str, float]] | None = None,
    top_k: int = 20,
    min_shared_tokens: int = 1,
) -> dict[str, list[str]]:
    """candidates_for_record applied to every Source 1 entity."""
    return {
        s1_id: candidates_for_record(
            tokens, *indexes, weights=weights, top_k=top_k, min_shared_tokens=min_shared_tokens
        )
        for s1_id, tokens in zip(s1_ids, s1_token_lists)
    }
