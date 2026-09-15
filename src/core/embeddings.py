from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfEmbedder:
    """Embedding léger sans téléchargement de modèle."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), stop_words=None)
        self.matrix = None

    def fit(self, texts: list[str]) -> np.ndarray:
        self.matrix = self.vectorizer.fit_transform(texts)
        return self.matrix

    def encode(self, texts: list[str]):
        if self.matrix is None:
            raise RuntimeError("L'embedder doit être entraîné avec fit() avant encode().")
        return self.vectorizer.transform(texts)

    def similarity(self, query: str, docs_matrix=None) -> np.ndarray:
        if docs_matrix is None:
            docs_matrix = self.matrix
        q = self.encode([query])
        return cosine_similarity(q, docs_matrix)[0]


class SentenceTransformerEmbedder:
    """Lazy local semantic embedder used by the persistent Chroma backend."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers est requis pour le backend Chroma."
                ) from exc
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    local_files_only=True,
                )
            except Exception:
                try:
                    self._model = SentenceTransformer(self.model_name)
                except Exception as exc:
                    raise RuntimeError(
                        "Impossible de charger le modele d'embeddings "
                        f"{self.model_name}. Une premiere connexion est "
                        "necessaire pour le telechargement du modele."
                    ) from exc
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
