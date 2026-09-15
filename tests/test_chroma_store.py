from src.core.chroma_store import ChromaVectorStore
from src.core.text_splitter import Chunk


class FakeEmbedder:
    def encode(self, texts):
        return [
            [
                float("stakeholder" in text.lower()),
                float("interface" in text.lower()),
                float("risk" in text.lower()),
            ]
            for text in texts
        ]


def test_chroma_store_adds_and_queries_chunks(tmp_path):
    store = ChromaVectorStore(
        tmp_path / "chroma",
        collection_name="test_collection",
        embedder=FakeEmbedder(),
    )
    store.add_chunks(
        [
            Chunk(
                id="one",
                source="reference.pdf",
                page=1,
                text="Stakeholder needs define the operational concept.",
                section="Needs",
            ),
            Chunk(
                id="two",
                source="reference.pdf",
                page=2,
                text="Interfaces support operational flows.",
                section="Interfaces",
            ),
        ],
        reset=True,
    )

    results = store.query("stakeholder operational needs", top_k=1)

    assert store.count == 2
    assert len(results) == 1
    assert results[0]["source_document"] == "reference.pdf"
    assert results[0]["page"] == 1
    assert results[0]["retrieval_score"] > 0

