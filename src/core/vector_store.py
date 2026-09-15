from __future__ import annotations
from dataclasses import dataclass
from .text_splitter import Chunk
from .embeddings import TfidfEmbedder
import re

@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float

class LocalVectorStore:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.embedder = TfidfEmbedder()
        self.matrix = self.embedder.fit([c.text for c in chunks]) if chunks else None

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self.chunks or self.matrix is None:
            return []
        scores = self.embedder.similarity(query, self.matrix)
        query_tokens = set(re.findall(r"[A-Za-zÀ-ÿ]{4,}", query.lower()))
        candidates = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )[: max(top_k * 4, top_k)]
        reranked = []
        for index, cosine_score in candidates:
            chunk_tokens = set(
                re.findall(r"[A-Za-zÀ-ÿ]{4,}", self.chunks[index].text.lower())
            )
            coverage = (
                len(query_tokens & chunk_tokens) / len(query_tokens)
                if query_tokens
                else 0.0
            )
            combined_score = 0.75 * float(cosine_score) + 0.25 * coverage
            reranked.append((index, combined_score))
        reranked.sort(key=lambda item: item[1], reverse=True)

        selected = []
        page_counts: dict[tuple[str, int | None], int] = {}
        for index, score in reranked:
            page_key = (self.chunks[index].source, self.chunks[index].page)
            if page_counts.get(page_key, 0) >= 2:
                continue
            selected.append(RetrievedChunk(self.chunks[index], float(score)))
            page_counts[page_key] = page_counts.get(page_key, 0) + 1
            if len(selected) >= top_k:
                break
        return selected
