from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.core.pdf_loader import load_pdfs_from_folder
from src.core.text_splitter import Chunk, split_pages
from src.core.vector_store import LocalVectorStore, RetrievedChunk
from src.domain.rule_based_evaluator import related_rule_ids_for_text
from src.rules.rule_repository import load_official_rules_repository


RAG_DOMAIN_TERMS = (
    "ConOps OpsCon operational concept stakeholders requirements risks "
    "interfaces capabilities operational scenarios system of systems "
    "constituent systems operational services operational flows governance"
)

RULE_RAG_QUERIES = {
    "D1": "problem space current situation needs motivations change future outcomes",
    "D2": "solution space future operations system of systems operational realization",
    "C5": "stakeholder needs expected outcomes pain points desired improvements jobs to be done",
    "O5": "operational flows information material energy financial exchange",
    "O8": "interfaces communication links APIs interoperability flows exchanged information",
}


class RagEngine:
    def __init__(
        self,
        reference_folder: str | Path,
        *,
        backend: str | None = None,
        vector_store=None,
    ):
        self.reference_folder = Path(reference_folder)
        self.backend = (backend or settings.rag_backend).lower()
        self.vector_store = vector_store
        self.nb_chunks = (
            int(getattr(vector_store, "count", 0)) if vector_store else 0
        )
        self.retrieval_log: list[dict] = []
        if self.vector_store is None and self.backend == "chroma":
            self.vector_store = self._new_chroma_store()
            self.nb_chunks = self.vector_store.count

    @staticmethod
    def semantic_query(document_text: str) -> str:
        compact = " ".join(document_text.split())
        return f"{RAG_DOMAIN_TERMS}\nDocument context:\n{compact[:3000]}"

    @staticmethod
    def _new_chroma_store():
        from src.core.chroma_store import ChromaVectorStore

        return ChromaVectorStore(
            settings.chroma_path,
            settings.chroma_collection,
            settings.embedding_model,
        )

    @property
    def is_ready(self) -> bool:
        return self.nb_chunks > 0

    def build(self) -> int:
        pages = load_pdfs_from_folder(self.reference_folder)
        chunks = split_pages(
            pages,
            settings.chunk_size,
            settings.chunk_overlap,
        )
        if self.backend == "chroma":
            if self.vector_store is None:
                self.vector_store = self._new_chroma_store()
            self.nb_chunks = self.vector_store.add_chunks(chunks, reset=True)
        else:
            self.vector_store = LocalVectorStore(chunks)
            self.nb_chunks = len(chunks)
        return self.nb_chunks

    def _ensure_store(self) -> None:
        if self.vector_store is None:
            self.build()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        rule_id: str = "",
    ) -> list[RetrievedChunk]:
        self._ensure_store()
        requested_top_k = top_k or settings.rag_top_k
        if self.backend == "chroma":
            rows = self.vector_store.query(query, requested_top_k)
            results = [
                RetrievedChunk(
                    Chunk(
                        id=row["chunk_id"],
                        source=row["source_document"],
                        page=row["page"],
                        text=row["text"],
                        section=row["section"],
                        source_path=row["source_path"],
                    ),
                    row["retrieval_score"],
                )
                for row in rows
            ]
        else:
            results = (
                self.vector_store.search(query, requested_top_k)
                if self.vector_store
                else []
            )
            rows = [
                {
                    "chunk_id": result.chunk.id,
                    "text": result.chunk.text,
                    "source_document": result.chunk.source,
                    "page": result.chunk.page,
                    "section": result.chunk.section,
                    "retrieval_score": round(result.score, 6),
                    "distance": round(max(0.0, 1.0 - result.score), 6),
                    "source_path": result.chunk.source_path,
                }
                for result in results
            ]

        self.retrieval_log.extend(
            {
                "query": query,
                "rule_id": rule_id,
                "chunk_id": row["chunk_id"],
                "source_document": row["source_document"],
                "document_name": row["source_document"],
                "page": row["page"],
                "section": row["section"],
                "embedding_model": settings.embedding_model,
                "retrieval_score": row["retrieval_score"],
                "similarity_score": row["retrieval_score"],
                "distance": row["distance"],
                "used_in_prompt": True,
                "used_in_answer": False,
                "related_item_ids": [],
                "chunk_text": row["text"],
                "source_path": row["source_path"],
                "related_rule_ids": related_rule_ids_for_text(row["text"]),
            }
            for row in rows
        )
        return results

    def retrieve_for_prompt(
        self,
        document_text: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        return self.retrieve(self.semantic_query(document_text), top_k)

    def rule_query(self, rule: dict) -> str:
        rule_id = str(rule.get("id", ""))
        if rule_id in RULE_RAG_QUERIES:
            return RULE_RAG_QUERIES[rule_id]
        terms = [
            *rule.get("evidence_keywords", [])[:8],
            *rule.get("evaluation_questions", [])[:3],
        ]
        return " ".join(str(term) for term in terms if term) or rule.get(
            "name",
            rule_id,
        )

    def retrieve_for_rules(
        self,
        *,
        top_k_per_rule: int = 2,
        rule_ids: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        repository = load_official_rules_repository()
        retrieved: list[RetrievedChunk] = []
        for rule in repository.all():
            rule_id = str(rule["id"])
            if rule_ids and rule_id not in rule_ids:
                continue
            retrieved.extend(
                self.retrieve(
                    self.rule_query(rule),
                    top_k_per_rule,
                    rule_id=rule_id,
                )
            )
        return retrieved

    def context_for_prompt(
        self,
        query: str,
        top_k: int | None = None,
        *,
        expand_query: bool = False,
    ) -> str:
        effective_query = self.semantic_query(query) if expand_query else query
        results = self.retrieve(effective_query, top_k)
        if expand_query:
            self.retrieve_for_rules(top_k_per_rule=1)
        blocks = []
        for index, result in enumerate(results, start=1):
            page = result.chunk.page or "inconnue"
            blocks.append(
                f"[RAG-{index}] Source: {result.chunk.source}, "
                f"page: {page}, score: {result.score:.3f}\n"
                f"{result.chunk.text}"
            )
        return "\n---\n".join(blocks)

    def clear_retrieval_log(self) -> None:
        self.retrieval_log.clear()

    def retrieval_stats(self) -> dict:
        if not self.retrieval_log:
            return {
                "retrieved_chunks_count": 0,
                "mean_retrieval_score": 0.0,
                "top_chunk_source": "",
                "top_chunk_page": None,
                "rag_context_tokens": 0,
                "retrieval_hit_rate": 0.0,
                "chunks_used_in_answer": 0,
                "evidence_from_rag_count": 0,
                "rule_evidence_coverage_score": 0.0,
                "rag_traceability_score": 0.0,
            }
        top = max(
            self.retrieval_log,
            key=lambda row: row["retrieval_score"],
        )
        used_count = sum(
            bool(row.get("used_in_answer")) for row in self.retrieval_log
        )
        rule_ids = {
            row.get("rule_id")
            for row in self.retrieval_log
            if row.get("rule_id")
        }
        used_rule_ids = {
            row.get("rule_id")
            for row in self.retrieval_log
            if row.get("rule_id") and row.get("used_in_answer")
        }
        traceable_rows = sum(
            bool(row.get("chunk_id")) and bool(row.get("source_document"))
            for row in self.retrieval_log
        )
        return {
            "retrieved_chunks_count": len(self.retrieval_log),
            "mean_retrieval_score": round(
                sum(row["retrieval_score"] for row in self.retrieval_log)
                / len(self.retrieval_log),
                3,
            ),
            "top_chunk_source": top["source_document"],
            "top_chunk_page": top["page"],
            "rag_context_tokens": sum(
                len(row["chunk_text"].split()) for row in self.retrieval_log
            ),
            "retrieval_hit_rate": float(used_count > 0),
            "chunks_used_in_answer": used_count,
            "evidence_from_rag_count": used_count,
            "rule_evidence_coverage_score": round(
                len(used_rule_ids) / len(rule_ids),
                3,
            )
            if rule_ids
            else 0.0,
            "rag_traceability_score": round(
                traceable_rows / len(self.retrieval_log),
                3,
            ),
        }
