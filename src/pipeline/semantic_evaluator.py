from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def semantic_similarity(text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([text_a, text_b])
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
