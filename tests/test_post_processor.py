from src.knowledge.ontology import get_knowledge_context
from src.pipeline.quality import classify_risk_candidate, clean_requirement_text


def test_requirement_cleanup_and_risk_classification():
    cleaned = clean_requirement_text(
        "The system shall support improve trajectory prediction."
    )
    assert "shall enable improved" in cleaned
    assert classify_risk_candidate("Cyberattack and outage risk") == "risk"
    assert classify_risk_candidate("Backup redundancy solution") == "mitigation"


def test_knowledge_formats_have_increasing_structure():
    k1 = get_knowledge_context("K1")
    k2 = get_knowledge_context("K2")
    k3 = get_knowledge_context("K3")
    assert "Stakeholder" in k1
    assert "Definitions" in k2
    assert "--expresses-->" in k3
    assert "FutureOperationalSoS" in k3
