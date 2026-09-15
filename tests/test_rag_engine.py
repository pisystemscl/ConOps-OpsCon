from src.core.chroma_store import ChromaVectorStore
from src.core.pdf_loader import PageText
from src.core.rag_engine import RagEngine


class FakeEmbedder:
    def encode(self, texts):
        return [
            [
                float("opscon" in text.lower()),
                float("stakeholder" in text.lower()),
                float("interface" in text.lower()),
            ]
            for text in texts
        ]


def test_rag_engine_builds_and_returns_traceable_chunks(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "src.core.rag_engine.load_pdfs_from_folder",
        lambda folder: [
            PageText(
                source="opscon_reference.pdf",
                page=7,
                text=(
                    "OpsCon operational scenarios allocate interfaces and "
                    "services to constituent systems. " * 8
                ),
                source_path="/references/opscon_reference.pdf",
            )
        ],
    )
    store = ChromaVectorStore(
        tmp_path / "chroma",
        collection_name="rag_test",
        embedder=FakeEmbedder(),
    )
    engine = RagEngine(
        tmp_path,
        backend="chroma",
        vector_store=store,
    )

    count = engine.build()
    context = engine.context_for_prompt(
        "Passenger carrying OpsCon with interfaces",
        top_k=1,
        expand_query=True,
    )

    assert count > 0
    assert "Source: opscon_reference.pdf" in context
    assert "page: 7" in context
    assert engine.retrieval_log[0]["used_in_prompt"] is True

