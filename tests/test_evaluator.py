from src.core.schema import ConOpsAnalysis, Stakeholder, Requirement
from src.pipeline.evaluator import ConOpsEvaluator


def test_evaluator_basic():
    analysis = ConOpsAnalysis(
        document_title="Test",
        purpose_scope="Définir une vision future",
        general_context="Contexte",
        as_is=["système existant"],
        to_be=["service futur"],
        stakeholders=[Stakeholder(name="User", role="utilisateur")],
        existing_systems=["legacy system"],
        existing_services=["service actuel"],
        expected_services=["service futur"],
        gaps=["gap"],
        capabilities=["capability"],
        requirements=[Requirement(id="REQ-001", text="The system shall provide a future service")],
    )
    report = ConOpsEvaluator().evaluate(analysis, "User legacy system service futur gap capability", "C0")
    assert report.final_score >= 0
    assert report.verdict != "Excellent"
    assert report.human_review_rate > 0


def test_conops_review_does_not_penalize_missing_opscon_rules():
    analysis = ConOpsAnalysis(
        as_is=["Current services have limitations."],
        to_be=["A shared future vision."],
        stakeholders=[Stakeholder(name="Users")],
        expected_services=["Future authorization service"],
        gaps=["Missing rapid authorization capability"],
        capabilities=["Rapid authorization capability"],
    )
    source = (
        "The current situation includes current services and current "
        "limitations. Stakeholder needs and expected outcomes justify change. "
        "Future services remain black-box. Capability gaps, future "
        "capabilities, a shared future vision and regulatory constraints are "
        "described."
    )
    report = ConOpsEvaluator().evaluate(analysis, source, "C3")
    assert report.rule_based_classification == "ConOps"
    assert not any(
        recommendation.startswith(("O1 -", "O2 -", "O6 -"))
        for recommendation in report.recommendations
    )
