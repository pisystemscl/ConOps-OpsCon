from src.core.schema import ConOpsAnalysis, Evidence, Requirement
from src.sysml.sysml_graph_generator import save_sysml_graph


def test_sysml_graph_generator_exports_files(tmp_path):
    analysis = ConOpsAnalysis(
        requirements=[
            Requirement(
                id="REQ-001",
                text="The system shall share operational data.",
                evidence=[Evidence(text="Operational data is shared.", page=3)],
            )
        ]
    )

    paths = save_sysml_graph(analysis, tmp_path)

    assert paths["sysml_model"].exists()
    assert paths["sysml_graph"].exists()
    assert paths["sysml_traceability_matrix"].exists()
    assert paths["sysml_graph_metrics"].exists()
