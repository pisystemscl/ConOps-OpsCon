from src.core.rag_engine import RagEngine
from src.core.text_splitter import Chunk
from src.core.vector_store import LocalVectorStore


def test_rag_traceability_logs_rule_id_and_metrics(tmp_path):
    chunk = Chunk(
        id="chunk-1",
        source="ref.pdf",
        page=4,
        text="Stakeholder needs and expected outcomes are documented.",
        section="Section",
        source_path="ref.pdf",
    )
    engine = RagEngine(
        tmp_path,
        backend="local",
        vector_store=LocalVectorStore([chunk]),
    )

    results = engine.retrieve("stakeholder needs expected outcomes", rule_id="C5")
    engine.retrieval_log[0]["used_in_answer"] = True
    stats = engine.retrieval_stats()

    assert results
    assert engine.retrieval_log[0]["rule_id"] == "C5"
    assert engine.retrieval_log[0]["embedding_model"]
    assert stats["rag_traceability_score"] == 1.0
    assert stats["rule_evidence_coverage_score"] == 1.0
