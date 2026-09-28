import pandas as pd

from src.entity_resolution.fast_block import build_tfidf_matrix, generate_candidates_fast
from src.entity_resolution.pipeline import build_source_indexes, generate_combined_candidates


def _make_source(entity_ids, name_tokens, addr_tokens):
    return pd.DataFrame({
        "entity_id": entity_ids,
        "business_name": ["x"] * len(entity_ids),
        "business_address": ["x"] * len(entity_ids),
        "name_tokens": name_tokens,
        "addr_tokens": addr_tokens,
    })


def test_build_tfidf_matrix_handles_all_empty_documents():
    # A source with no address data at all (a real case in this dataset,
    # e.g. some Source 3 rows have blank addresses) — must not crash.
    _, matrix = build_tfidf_matrix([[], [], []])
    assert matrix.shape == (3, 0)


def test_generate_candidates_fast_matches_dict_based_ranking():
    # Purging disabled (max_doc_freq=1.0) to isolate ranking logic from
    # purge-threshold behavior, which doesn't make sense to test on a
    # handful of documents anyway (2% of 3 docs rounds to under 1).
    s1 = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2", "S1-3"],
        "name_tokens": [["acme", "corp"], ["widget"], ["nomatch"]],
        "addr_tokens": [["main", "st"], ["oak", "ave"], ["nowhere"]],
    })
    s2 = _make_source(
        ["S2-1", "S2-2", "S2-3"],
        [["acme", "corp"], ["acme"], ["other"]],
        [["main", "st"], ["main"], ["nowhere"]],
    )
    s3 = _make_source(["S3-1"], [["widget"]], [[]])

    s2_indexes = build_source_indexes(s2, name_max_doc_freq=1.0, addr_max_doc_freq=1.0)
    s3_indexes = build_source_indexes(s3, name_max_doc_freq=1.0, addr_max_doc_freq=1.0)
    old = generate_combined_candidates(s1, s2_indexes, s3_indexes, top_k=10)
    new = generate_candidates_fast(s1, [s2, s3], top_k=10, name_max_doc_freq=1.0, addr_max_doc_freq=1.0)

    for s1_id in old:
        assert set(old[s1_id]) == set(new[s1_id]), f"mismatch on {s1_id}: {old[s1_id]} vs {new[s1_id]}"


def test_generate_candidates_fast_respects_top_k():
    s1 = pd.DataFrame({"entity_id": ["S1-1"], "name_tokens": [["common"]], "addr_tokens": [[]]})
    s2 = _make_source(
        [f"S2-{i}" for i in range(5)],
        [["common"] for _ in range(5)],
        [[] for _ in range(5)],
    )
    result = generate_candidates_fast(s1, [s2], top_k=2, name_max_doc_freq=1.0)
    assert len(result["S1-1"]) == 2


def test_generate_candidates_fast_empty_query_gives_no_candidates():
    s1 = pd.DataFrame({"entity_id": ["S1-1"], "name_tokens": [[]], "addr_tokens": [[]]})
    s2 = _make_source(["S2-1"], [["acme"]], [["main"]])
    result = generate_candidates_fast(s1, [s2], top_k=10)
    assert result["S1-1"] == []


def test_generate_candidates_fast_pools_across_multiple_sources():
    s1 = pd.DataFrame({
        "entity_id": ["S1-1"],
        "name_tokens": [["acme"]],
        "addr_tokens": [[]],
    })
    s2 = _make_source(["S2-1"], [["acme"]], [[]])
    s3 = _make_source(["S3-1"], [["acme"]], [[]])
    result = generate_candidates_fast(s1, [s2, s3], top_k=10, name_max_doc_freq=1.0)
    assert set(result["S1-1"]) == {"S2-1", "S3-1"}
