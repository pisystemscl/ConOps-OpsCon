from src.core.chroma_store import ChromaVectorStore
from src.core.text_splitter import Chunk
from src.core.rag_engine import RagEngine
from src.llm.providers import MockLLM
from src.pipeline.experiment_runner import ExperimentRunner


class FakeEmbedder:
    def encode(self, texts):
        return [
            [
                float("stakeholder" in text.lower()),
                float("interface" in text.lower()),
                float("operational" in text.lower()),
            ]
            for text in texts
        ]


def test_c4_exposes_non_empty_rag_metrics(monkeypatch, tmp_path):
    store = ChromaVectorStore(
        tmp_path / "chroma",
        collection_name="metrics_test",
        embedder=FakeEmbedder(),
    )
    store.add_chunks(
        [
            Chunk(
                id="chunk-1",
                source="reference.pdf",
                page=9,
                text=(
                    "Operational interfaces support stakeholder services "
                    "and operational flows."
                ),
                section="Operational interfaces",
            )
        ],
        reset=True,
    )
    rag = RagEngine(tmp_path, backend="chroma", vector_store=store)
    monkeypatch.setattr(
        "src.pipeline.experiment_runner.get_llm_for_configuration",
        lambda code, provider=None, model=None: MockLLM(),
    )
    runner = ExperimentRunner(provider="mock", rag=rag)

    results, _, _ = runner.run(
        "Stakeholders require operational interfaces and resilient services.",
        selected_codes=["C4"],
        document_name="target.pdf",
    )

    result = results[0]
    assert result.rag_used is True
    assert result.retrieved_chunks_count > 0
    assert result.rag_usage_score > 0
    assert result.top_chunk_source == "reference.pdf"
    assert result.top_chunk_page == 9
    assert runner.rag_rows
