from src.core.schema import ConOpsAnalysis, Requirement, Stakeholder
from src.graph.graph_generator import build_graph, graph_data, graph_metrics


def test_knowledge_graph_exports_metrics_and_data():
    analysis = ConOpsAnalysis(
        document_title="Sample",
        document_classification="ConOps",
        stakeholders=[Stakeholder(name="Airspace Users")],
        stakeholder_needs=["Airspace Users need resilient services"],
        requirements=[
            Requirement(
                id="REQ-001",
                text="The system shall support resilient services.",
            )
        ],
    )

    graph = build_graph(analysis)
    metrics = graph_metrics(graph)
    data = graph_data(graph)

    assert metrics["nodes_count"] > 0
    assert metrics["edges_count"] > 0
    assert data["nodes"]
    assert data["edges"]
