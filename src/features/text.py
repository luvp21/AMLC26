"""Text featurization: a cheap TF-IDF path for the day-1 baseline, and a
sentence-embedding path (MiniLM/DistilBERT-class) for the real model, matching
what 2023's and 2025's top teams used before fine-tuning anything.
"""
from __future__ import annotations

import numpy as np


class TfidfFeaturizer:
    """Fit once on train text, transform train/test consistently."""

    def __init__(self, max_features: int = 50_000, ngram_range: tuple[int, int] = (1, 2)):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self._fitted = False

    def fit_transform(self, texts: list[str]):
        self._fitted = True
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: list[str]):
        if not self._fitted:
            raise RuntimeError("call fit_transform on train text first")
        return self.vectorizer.transform(texts)


def embed_texts(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: str | None = None,
) -> np.ndarray:
    """Dense sentence embeddings for downstream GBM features or as a fusion
    input alongside image embeddings. Lazy-imports sentence-transformers so
    importing this module doesn't require torch unless you actually call this.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
    )
    return embeddings


def basic_text_stats(texts: list[str]) -> "np.ndarray":
    """Cheap structural features: length, word count, digit ratio, etc.
    Surprisingly useful alongside embeddings for regression targets like
    price or length that correlate with description verbosity/pack-quantity
    mentions.
    """
    feats = []
    for t in texts:
        t = t or ""
        n_chars = len(t)
        n_words = len(t.split())
        n_digits = sum(c.isdigit() for c in t)
        n_upper = sum(c.isupper() for c in t)
        feats.append([
            n_chars,
            n_words,
            n_digits / max(n_chars, 1),
            n_upper / max(n_chars, 1),
            t.count(","),
            t.count("x") + t.count("X"),  # dimension-style "10 x 20" mentions
        ])
    return np.array(feats, dtype=float)
