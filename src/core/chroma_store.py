from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from src.core.embeddings import SentenceTransformerEmbedder
from src.core.text_splitter import Chunk


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class ChromaVectorStore:
    """Persistent local Chroma collection for reference-document chunks."""

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str = "conops_reference_docs",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        embedder: Embedder | None = None,
    ):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "ChromaDB est requis par requirements.txt."
            ) from exc

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = embedder or SentenceTransformerEmbedder(
            embedding_model
        )

    @staticmethod
    def _stable_id(chunk: Chunk, index: int) -> str:
        payload = (
            f"{chunk.source_path}|{chunk.source}|{chunk.page}|"
            f"{index}|{chunk.text}"
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        return f"chunk-{digest}"

    @property
    def count(self) -> int:
        return int(self.collection.count())

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except ValueError:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], *, reset: bool = False) -> int:
        if reset:
            self.reset()
        if not chunks:
            return self.count

        ids = [self._stable_id(chunk, index) for index, chunk in enumerate(chunks)]
        documents = [chunk.text for chunk in chunks]
        embeddings = self.embedder.encode(documents)
        metadatas = [
            {
                "document_name": chunk.source,
                "page": int(chunk.page or 0),
                "section": chunk.section or "",
                "source_path": chunk.source_path or "",
                "chunk_index": index,
                "original_chunk_id": chunk.id,
            }
            for index, chunk in enumerate(chunks)
        ]
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return self.count

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        if self.count == 0:
            return []
        query_embedding = self.embedder.encode([query_text])
        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.count),
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        rows = []
        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            distance_value = float(distance)
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "source_document": metadata.get("document_name", ""),
                    "page": metadata.get("page") or None,
                    "section": metadata.get("section", ""),
                    "retrieval_score": round(
                        max(0.0, min(1.0, 1.0 - distance_value)),
                        6,
                    ),
                    "distance": round(distance_value, 6),
                    "source_path": metadata.get("source_path", ""),
                    "chunk_index": metadata.get("chunk_index"),
                }
            )
        return rows
