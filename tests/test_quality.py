from src.core.schema import ConOpsAnalysis
from src.domain.rule_based_evaluator import evaluate_document_rules
from src.pipeline.evaluator import ConOpsEvaluator
from src.graph.graph_generator import build_graph
from src.pipeline.post_processor import enrich_analysis
from src.pipeline.quality import (
    classify_risk_candidate,
    clean_requirement_text,
    find_page_for_evidence,
    get_verdict,
    normalize_actor_name,
)
from src.pipeline.sysml_generator import generate_sysml_v2


def test_quality_classifiers_separate_driver_mitigation_and_risk():
    assert classify_risk_candidate("The network must handle 50 000 flights") == "driver"
    assert classify_risk_candidate("ATC centres provide backup redundancy") == "mitigation"
    assert classify_risk_candidate("Cyberattack and outage risk") == "risk"


def test_requirement_cleanup_repairs_bad_grammar():
    cleaned = clean_requirement_text(
        "The system shall support improve traffic demand prediction."
    )
    assert cleaned.startswith("The network shall enable improved")


def test_actor_aliases_are_normalized_in_graph():
    analysis = ConOpsAnalysis.model_validate(
        {
            "stakeholders": [{"name": "ANSP"}],
            "interfaces": [
                {
                    "source": "ANSPs",
                    "target": "NM",
                    "exchanged_information": "Capacity data",
                }
            ],
        }
    )
    graph = build_graph(analysis)
    labels = [attributes.get("label") for _, attributes in graph.nodes(data=True)]
    assert labels.count("ANSPs") == 1
    assert normalize_actor_name("NM") == "Network Manager"


def test_official_rules_v3_are_evaluable():
    source = (
        "[Page 4]\nThe current situation describes stakeholder needs and "
        "motivation for change.\n[Page 8]\nFuture services are expected."
    )
    result = evaluate_document_rules(source, ConOpsAnalysis())
    assert result["rules_version"] == "3.0"
    assert result["classification_by_rules"] in {
        "ConOps",
        "OpsCon",
        "Hybrid",
        "Neither",
    }
    assert result["rule_assessment_detailed"]


def test_evidence_page_is_recovered_from_page_markers():
    source = (
        "[Page 1]\nGeneral introduction.\n\n"
        "[Page 2]\nAirspace Users provide 4D trajectories via FF-ICE/eFPL."
    )
    assert (
        find_page_for_evidence(
            "Airspace Users provide 4D trajectories via FF-ICE/eFPL.",
            source,
        )
        == 2
    )


def test_atm_enrichment_adds_stakeholders_interfaces_and_pages():
    source = (
        "[Page 11]\nAirspace Users provide 4D business trajectories via "
        "FF-ICE/eFPL to the Network Manager. ANSPs exchange real-time data "
        "with the Network Manager through SWIM."
    )
    analysis = enrich_analysis(ConOpsAnalysis(), source)
    names = {item.name for item in analysis.stakeholders}
    assert {"Airspace Users", "ANSPs", "Network Manager"} <= names
    assert analysis.interfaces
    assert analysis.interfaces[0].evidence[0].page == 11


def test_false_risks_are_reclassified():
    analysis = ConOpsAnalysis.model_validate(
        {
            "risks": [
                {"description": "The Network shall maintain safety performance."},
                {
                    "description": (
                        "Cloud based cyber-secure solutions offer scalability."
                    )
                },
                {"description": "Cyberattack and system outage may disrupt service."},
            ]
        }
    )
    enriched = enrich_analysis(analysis, "[Page 1]\nCyberattack and system outage may disrupt service.")
    assert len(enriched.risks) == 1
    assert enriched.mitigations
    assert enriched.constraints


def test_verdict_requires_quality_not_only_final_score():
    assert (
        get_verdict(88, human_review_rate=0.70, classification_score=0.65)
        == "Prometteur - revue humaine nécessaire"
    )


def test_sysml_uses_black_box_future_service_for_conops():
    analysis = ConOpsAnalysis(expected_services=["Trajectory Management"])
    sysml = generate_sysml_v2(analysis)
    assert "part def FutureService;" in sysml
    assert ": FutureService;" in sysml
    assert "FutureOperationalSoS" not in sysml
