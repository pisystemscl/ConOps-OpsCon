from src.core.schema import ConOpsAnalysis, Evidence, Requirement
from src.graph.graph_generator import build_graph, graph_metrics


def test_graph_grounding_metrics_count_grounded_nodes():
    analysis = ConOpsAnalysis(
        requirements=[
            Requirement(
                id="REQ-001",
                text="The system shall share operational data.",
                evidence=[
                    Evidence(
                        text="Operational data shall be shared.",
                        page=7,
                        source="target.pdf",
                    )
                ],
            )
        ]
    )

    graph = build_graph(analysis)
    metrics = graph_metrics(graph)
    req_node = graph.nodes["REQ-001"]

    assert req_node["source_page"] == 7
    assert req_node["evidence_quote"]
    assert req_node["source_document"] == "target.pdf"
    assert metrics["grounded_nodes_count"] >= 1
    assert metrics["ungrounded_nodes_count"] >= 0
