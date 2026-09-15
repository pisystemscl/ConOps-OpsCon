from pathlib import Path

from src.domain.ontology_builder import build_ontology
from src.domain.rule_based_evaluator import (
    evaluate_document_rules,
    score_assessments,
)
from src.domain.rules_loader import (
    EXPECTED_RULE_IDS,
    RULES_PATH,
    all_rules,
    load_rules,
    validate_rules_integrity,
)


def test_rules_file_and_ontology_exclude_ambiguous_concepts():
    rules_text = Path(RULES_PATH).read_text(encoding="utf-8").lower()
    assert "ecosystem" not in rules_text
    ontology = build_ontology()
    concepts = (
        ontology["conops"]["concepts"]
        + ontology["opscon"]["concepts"]
    )
    assert "Ecosystem" not in concepts
    assert "Value" not in concepts
    assert not any("SoS" in concept for concept in ontology["conops"]["concepts"])
    assert any("SoS" in concept for concept in ontology["opscon"]["concepts"])


def test_all_official_v3_rule_ids_exist():
    rules = load_rules()
    assert validate_rules_integrity(rules)
    identifiers = {rule["id"] for rule in all_rules(rules)}
    assert EXPECTED_RULE_IDS <= identifiers


def test_c9_and_c10_are_critical_forbidden_rules():
    by_id = {rule["id"]: rule for rule in all_rules()}
    for identifier in ("C9", "C10"):
        assert by_id[identifier]["forbidden"] is True
        assert by_id[identifier]["critical"] is True


def test_forbidden_violation_penalty_exceeds_missing_expected_penalty():
    result = evaluate_document_rules(
        "The architecture view defines constituent systems and SoS structure."
    )
    by_id = {
        item["rule_id"]: item
        for item in result["rule_assessment_detailed"]
    }
    assert by_id["C9"]["status"] == "forbidden_present"
    assert abs(by_id["C9"]["score_contribution"]) > abs(
        by_id["C1"]["score_contribution"]
    )


def test_classification_is_hybrid_when_both_rule_families_are_high():
    rules = load_rules()
    assessments = []
    for rule in all_rules(rules):
        status = (
            "forbidden_absent"
            if rule.get("forbidden")
            else "present"
        )
        assessments.append(
            {
                "rule_id": rule["id"],
                "rule_name": rule["name"],
                "rule_family": rule["family"],
                "expected": rule.get(
                    "expected",
                    rule.get("expected_for", []),
                ),
                "forbidden": bool(rule.get("forbidden")),
                "weight": float(rule["weight"]),
                "status": status,
                "matched_terms": [],
                "evidence_quote": "Validated evidence.",
                "source_page": 1,
                "comment": rule["description"],
                "human_review_needed": False,
            }
        )
    result = score_assessments(assessments, rules)
    assert result["classification_by_rules"] == "Hybrid"
