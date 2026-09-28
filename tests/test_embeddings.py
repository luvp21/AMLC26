import numpy as np
import pandas as pd

from src.entity_resolution.embeddings import cosine_similarity_column
from src.entity_resolution.features import EMBEDDING_FEATURE_NAMES, FEATURE_NAMES, build_full_feature_matrix


def test_cosine_similarity_column_identical_vectors_gives_one():
    lookup = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0])}
    result = cosine_similarity_column(["a"], ["b"], lookup)
    assert abs(result[0] - 1.0) < 1e-6


def test_cosine_similarity_column_orthogonal_vectors_gives_zero():
    lookup = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    result = cosine_similarity_column(["a"], ["b"], lookup)
    assert abs(result[0]) < 1e-6


def test_cosine_similarity_column_missing_text_defaults_to_zero_vector():
    lookup = {"a": np.array([1.0, 0.0])}
    result = cosine_similarity_column(["a"], ["never_embedded"], lookup)
    assert result[0] == 0.0


def test_cosine_similarity_column_empty_lookup_returns_zeros():
    result = cosine_similarity_column(["a", "b"], ["c", "d"], {})
    assert list(result) == [0.0, 0.0]


def test_build_full_feature_matrix_without_embeddings_matches_base_feature_names():
    pairs_df = pd.DataFrame({
        "s1_name": ["acme"], "s1_addr": ["main st"],
        "s1_name_tokens": [["acme"]], "s1_addr_tokens": [["main", "st"]], "s1_country": ["US"],
        "c_name": ["acme corp"], "c_addr": ["main st"],
        "c_name_tokens": [["acme", "corp"]], "c_addr_tokens": [["main", "st"]], "c_country": ["US"],
    })
    X, names = build_full_feature_matrix(pairs_df, use_embeddings=False)
    assert names == FEATURE_NAMES
    assert X.shape == (1, len(FEATURE_NAMES))


def test_build_full_feature_matrix_with_embeddings_appends_columns(monkeypatch):
    pairs_df = pd.DataFrame({
        "s1_name": ["acme"], "s1_addr": ["main st"],
        "s1_name_tokens": [["acme"]], "s1_addr_tokens": [["main", "st"]], "s1_country": ["US"],
        "c_name": ["acme corp"], "c_addr": ["main st"],
        "c_name_tokens": [["acme", "corp"]], "c_addr_tokens": [["main", "st"]], "c_country": ["US"],
    })

    def fake_build_embedding_features(df, model_name=None):
        return np.ones((len(df), 2), dtype=np.float32) * 0.5

    import src.entity_resolution.embeddings as embeddings_mod
    monkeypatch.setattr(embeddings_mod, "build_embedding_features", fake_build_embedding_features)

    X, names = build_full_feature_matrix(pairs_df, use_embeddings=True)
    assert names == FEATURE_NAMES + EMBEDDING_FEATURE_NAMES
    assert X.shape == (1, len(FEATURE_NAMES) + 2)
    assert list(X[0, -2:]) == [0.5, 0.5]


def test_build_feature_matrix_fast_is_bit_identical_to_per_row():
    from src.entity_resolution.features import build_feature_matrix, build_feature_matrix_fast

    pairs_df = pd.DataFrame({
        "s1_name": ["Acme Corp", "Widget Inc", "", "Bharat Traders"],
        "s1_addr": ["12 Main St", "", "", "MG Road, Pune"],
        "s1_name_tokens": [["acme"], ["widget"], [], ["bharat", "traders"]],
        "s1_addr_tokens": [["12", "main", "street"], [], [], ["mg", "road", "pune"]],
        "s1_country": ["US", "US", "India", "India"],
        "c_name": ["ACME Corporation", "Widgets", "", "Bharat Trader"],
        "c_addr": ["12 Main Street", "5 Oak", "", "M.G. Road Pune"],
        "c_name_tokens": [["acme"], ["widgets"], [], ["bharat", "trader"]],
        "c_addr_tokens": [["12", "main", "street"], ["oak"], [], ["mg", "road", "pune"]],
        "c_country": ["US", "India", "India", "India"],
    })
    expected = build_feature_matrix(pairs_df)
    assert np.array_equal(build_feature_matrix_fast(pairs_df, n_workers=1), expected)
    assert np.array_equal(build_feature_matrix_fast(pairs_df, n_workers=2, chunk_size=2), expected)
