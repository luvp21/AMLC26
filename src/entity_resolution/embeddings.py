"""Cross-script name similarity via multilingual sentence embeddings.

Motivation: the biggest unfixed gap identified for entity resolution is
Source 2's Devanagari business names — `anyascii` romanizes them phonetically
("प्राइम" -> "praim"), which doesn't match Source 1's actual English spelling
("Prime"). Token/string-similarity features (rapidfuzz, Jaccard) structurally
can't see these as similar, no matter how the tokens are weighted. Research
confirms multilingual sentence embeddings are the established fix for this
exact cross-lingual entity-matching problem (see docs/ENTITY_RESOLUTION_APPROACH.md
for citations) — a model like LaBSE or paraphrase-multilingual-MiniLM maps
"Prime" and "प्राइम" close together in embedding space based on meaning, not
surface characters.

This is a classifier FEATURE (cosine similarity of name/address embeddings),
not a blocking signal — blocking recall for cross-script names is already
handled by falling back to address tokens (see pipeline.py's docstring).
This targets classifier PRECISION on the candidates that already survive
blocking, particularly borderline India pairs.

Deliberately batches and dedupes: naively re-embedding a name for every
candidate PAIR it appears in would redundantly re-run the model millions of
times. Embedding each UNIQUE string once, then doing plain numpy cosine
similarity per pair, is orders of magnitude cheaper and is what makes this
tractable at full dataset scale.

Model download note: PARAM Shavak's network resets large binary downloads
from Hugging Face's CDN (see infra/param_shavak/README.md) — download the
model on a laptop first and rsync `~/.cache/huggingface/` over, then run
with HF_HUB_OFFLINE=1, exactly as documented there for other models.

Model choice — empirically checked, not assumed (25 Sept, real business
name pairs from the training data): tried both a general-purpose
multilingual paraphrase model and LaBSE. LaBSE wins decisively:

    model                                        true-match sim   unrelated sim
    paraphrase-multilingual-MiniLM-L12-v2 (~470MB)     0.324           0.155
    LaBSE (~1.8GB)                                     0.885-0.933     0.148-0.160

LaBSE's ~6x separation vs. the smaller model's ~2x makes it the clear
choice despite the larger download — it's Apache-2.0 licensed and
Google-published, purpose-built for exactly this cross-lingual bitext/entity
matching task (unlike general paraphrase models, which aren't well
calibrated for short proper-noun strings that are transliterations rather
than semantic translations).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# LaBSE (Language-Agnostic BERT Sentence Embedding, Apache-2.0, ~1.8GB):
# empirically confirmed (see module docstring) to give ~6x similarity
# separation between true cross-script matches and unrelated businesses,
# vs. ~2x for a general-purpose multilingual paraphrase model of similar
# size class. Purpose-built for cross-lingual matching, not just multilingual.
DEFAULT_MODEL = "sentence-transformers/LaBSE"


def embed_unique_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """Embed each distinct string in `texts` exactly once, return a
    text -> L2-normalized embedding lookup. Empty strings map to a zero
    vector (cosine similarity with anything is then 0, the sane default for
    "no address"/"no name" — never crashes, never falsely looks similar)."""
    from sentence_transformers import SentenceTransformer

    unique_texts = sorted({t for t in texts if t})
    if not unique_texts:
        return {}

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        unique_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so cosine similarity is just a dot product
    )
    lookup = dict(zip(unique_texts, embeddings))
    dim = embeddings.shape[1]
    lookup[""] = np.zeros(dim, dtype=embeddings.dtype)
    return lookup


def cosine_similarity_column(
    left_texts: list[str], right_texts: list[str], lookup: dict[str, np.ndarray]
) -> np.ndarray:
    """Vectorized cosine similarity for parallel lists of (left, right) text
    pairs, using a precomputed embed_unique_texts lookup — no per-pair model
    calls. Embeddings are already L2-normalized, so this is just a row-wise
    dot product."""
    if not lookup:
        return np.zeros(len(left_texts), dtype=np.float32)
    dim = next(iter(lookup.values())).shape[0]
    left = np.array([lookup.get(t, np.zeros(dim)) for t in left_texts], dtype=np.float32)
    right = np.array([lookup.get(t, np.zeros(dim)) for t in right_texts], dtype=np.float32)
    return np.sum(left * right, axis=1)


def build_embedding_features(
    pairs_df: pd.DataFrame, model_name: str = DEFAULT_MODEL, device: str | None = None
) -> np.ndarray:
    """pairs_df must have s1_name, c_name, s1_addr, c_addr columns (same
    convention as features.build_feature_matrix). Returns an (n, 2) float32
    array: [name_embedding_cosine, addr_embedding_cosine] — meant to be
    concatenated onto the rapidfuzz/Jaccard feature matrix, not replace it.
    """
    all_names = list(pairs_df.s1_name) + list(pairs_df.c_name)
    all_addrs = list(pairs_df.s1_addr) + list(pairs_df.c_addr)
    name_lookup = embed_unique_texts(all_names, model_name=model_name, device=device)
    addr_lookup = embed_unique_texts(all_addrs, model_name=model_name, device=device)

    name_sim = cosine_similarity_column(list(pairs_df.s1_name), list(pairs_df.c_name), name_lookup)
    addr_sim = cosine_similarity_column(list(pairs_df.s1_addr), list(pairs_df.c_addr), addr_lookup)
    return np.column_stack([name_sim, addr_sim]).astype(np.float32)
