import pandas as pd

from src.entity_resolution.block import (
    build_inverted_index,
    candidates_for_record,
    compute_idf_weights,
    generate_candidates,
)
from src.entity_resolution.normalize import normalize_address, normalize_name, tokenize
from src.entity_resolution.pipeline import generate_combined_candidates, generate_combined_candidates_parallel


def test_normalize_name_strips_suffix_and_lowercases():
    assert normalize_name("Sharma Traders Pvt Ltd") == "sharma traders"
    assert normalize_name("ACME Corp") == "acme"


def test_normalize_name_transliterates_devanagari():
    # "Ram Marketing" written in Devanagari — must not crash, must produce
    # ASCII-only output so it's comparable to Latin-script tokens.
    result = normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड")
    assert result.isascii()
    assert result != ""


def test_normalize_name_handles_empty_and_none():
    assert normalize_name("") == ""
    assert normalize_name(None) == ""
    assert normalize_name(float("nan")) == ""


def test_normalize_address_expands_common_abbreviations():
    result = normalize_address("123 Main St, Apt 4")
    assert "street" in result
    assert "apartment" in result


def test_tokenize_drops_single_char_tokens():
    assert tokenize("a bb ccc") == ["bb", "ccc"]


def test_tokenize_empty_string():
    assert tokenize("") == []


def test_build_inverted_index_purges_high_frequency_tokens():
    ids = [f"id{i}" for i in range(100)]
    # "common" appears in every record — should be purged at max_doc_freq=0.5
    token_lists = [["common", f"unique{i}"] for i in range(100)]
    index = build_inverted_index(ids, token_lists, max_doc_freq=0.5)
    assert "common" not in index
    assert "unique0" in index
    assert index["unique0"] == ["id0"]


def test_candidates_for_record_ranks_by_shared_token_count():
    index = {
        "acme": ["A", "B", "C"],
        "corp": ["A", "B"],
        "widgets": ["A"],
    }
    # querying with all three tokens: A shares 3, B shares 2, C shares 1
    result = candidates_for_record(["acme", "corp", "widgets"], index, top_k=10)
    assert result == ["A", "B", "C"]


def test_candidates_for_record_respects_top_k():
    index = {"tok": ["A", "B", "C", "D"]}
    result = candidates_for_record(["tok"], index, top_k=2)
    assert len(result) == 2


def test_candidates_for_record_min_shared_tokens_filters_weak_matches():
    index = {"a": ["X"], "b": ["Y"]}
    result = candidates_for_record(["a", "b"], index, top_k=10, min_shared_tokens=2)
    # X and Y each share only 1 token with the query — neither meets the threshold
    assert result == []


def test_compute_idf_weights_rarer_token_gets_higher_weight():
    index = {"common": ["A", "B", "C", "D"], "rare": ["A"]}
    weights = compute_idf_weights(index, n_docs=4)
    assert weights["rare"] > weights["common"]


def test_candidates_for_record_weighted_ranking_beats_raw_count():
    # A shares only the rare, distinctive token with the query. B shares two
    # generic connector tokens. Raw-count ranking would put B first (count=2
    # vs 1) — exactly the failure mode measured on the full-scale run
    # (docs/ENTITY_RESOLUTION_APPROACH.md: 97% of misses were "ranked out"
    # by ties among generic shared words). IDF weighting must put A first.
    index = {
        "vision": ["A"],
        "no": ["A", "B", "C", "D", "E", "F", "G", "H"],
        "city": ["A", "B", "C", "D", "E", "F", "G", "H"],
    }
    weights = compute_idf_weights(index, n_docs=8)
    result = candidates_for_record(["vision", "no", "city"], index, weights=[weights], top_k=10)
    assert result[0] == "A"


def test_candidates_for_record_without_weights_falls_back_to_raw_count():
    index = {"a": ["X"], "b": ["Y"], "c": ["Y"]}
    result = candidates_for_record(["a", "b", "c"], index, top_k=10)
    assert result[0] == "Y"  # 2 shared tokens vs X's 1, no weights supplied


def test_generate_candidates_multiple_entities():
    index = {"acme": ["S2-1"], "corp": ["S2-1", "S2-2"]}
    result = generate_candidates(
        ["S1-1", "S1-2"], [["acme", "corp"], ["corp"]], index, top_k=10
    )
    assert result["S1-1"] == ["S2-1", "S2-2"]
    assert result["S1-2"] == ["S2-1", "S2-2"]


def test_generate_combined_candidates_use_weights_false_matches_raw_count():
    # Protects the submission generator: it must reproduce the exact
    # raw-count ranking the classifier was trained/validated against, not
    # the newer IDF-weighted ranking, until retrained on the new candidates.
    s1 = pd.DataFrame({
        "entity_id": ["S1-1"],
        "name_tokens": [["vision", "no", "city"]],
        "addr_tokens": [[]],
    })
    # "distinctive" shares only the rare token "vision" (count=1).
    # "common" shares two generic tokens "no"+"city" (count=2), each common
    # to 8 records. Raw count favors "common" (2>1); IDF weighting must
    # favor "distinctive" since "vision" is far rarer than "no"/"city".
    name_index = {
        "vision": ["distinctive"],
        "no": ["common", "x", "y", "z", "w", "v", "u", "t"],
        "city": ["common", "x", "y", "z", "w", "v", "u", "t"],
    }
    from src.entity_resolution.block import compute_idf_weights

    weights = compute_idf_weights(name_index, n_docs=8)
    source = (name_index, {}, weights, {})

    weighted = generate_combined_candidates(s1, source, top_k=1, use_weights=True)
    unweighted = generate_combined_candidates(s1, source, top_k=1, use_weights=False)

    assert weighted["S1-1"] == ["distinctive"]
    assert unweighted["S1-1"] == ["common"]  # shares 2 generic tokens vs distinctive's 1


def test_parallel_candidate_generation_matches_serial():
    s1 = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2", "S1-3"],
        "name_tokens": [["acme", "corp"], ["widget"], ["nomatch"]],
        "addr_tokens": [["main", "st"], ["oak", "ave"], ["nowhere"]],
    })
    s2_indexes = (
        {"acme": ["S2-1"], "corp": ["S2-1", "S2-2"]},
        {"main": ["S2-1"], "st": ["S2-1", "S2-3"]},
        {},  # name weights (empty -> falls back to weight 1.0 per token)
        {},  # addr weights
    )
    s3_indexes = ({"widget": ["S3-1"]}, {}, {}, {})

    serial = generate_combined_candidates(s1, s2_indexes, s3_indexes, top_k=10)
    parallel = generate_combined_candidates_parallel(s1, s2_indexes, s3_indexes, top_k=10, n_workers=2)
    assert serial == parallel
