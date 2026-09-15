from pathlib import Path

from src.domain.ontology_builder import build_ontology
from src.domain.rule_based_evaluator import compute_rule_score
from src.domain.rules_loader import (
    EXPECTED_RULE_IDS,
    RULES_PATH,
    all_rules,
    load_rules,
    validate_rules_integrity,
)
from src.llm.prompts import build_extraction_prompt


def test_exact_official_rule_identifiers_exist():
    rules = load_rules()
    assert validate_rules_integrity(rules)
    assert {rule["id"] for rule in all_rules(rules)} == EXPECTED_RULE_IDS


def test_d1_and_d2_exist():
    identifiers = {rule["id"] for rule in all_rules()}
    assert {"D1", "D2"} <= identifiers


def test_c1_to_c14_exist():
    identifiers = {rule["id"] for rule in all_rules()}
    assert {f"C{index}" for index in range(1, 15)} <= identifiers


def test_o1_to_o14_and_o6a_exist():
    identifiers = {rule["id"] for rule in all_rules()}
    assert {f"O{index}" for index in range(1, 15)} <= identifiers
    assert "O6a" in identifiers


def test_prohibited_concepts_are_absent_from_rules_and_ontology():
    rules_text = Path(RULES_PATH).read_text(encoding="utf-8").lower()
    assert "ecosystem" not in rules_text
    ontology = build_ontology()
    concepts = ontology["conops"]["concepts"] + ontology["opscon"]["concepts"]
    assert "Ecosystem" not in concepts
    assert "Value" not in concepts
    prompt = build_extraction_prompt(
        "Target document.",
        use_template=True,
        use_rules=True,
    )
    assert "ecosystem" not in prompt.lower()


def test_system_of_systems_is_opscon_only():
    ontology = build_ontology()
    assert not any("SoS" in item for item in ontology["conops"]["concepts"])
    assert any("SoS" in item for item in ontology["opscon"]["concepts"])


def test_c9_and_c10_are_forbidden_conops_rules():
    rules_by_id = {rule["id"]: rule for rule in all_rules()}
    assert rules_by_id["C9"]["forbidden"] is True
    assert rules_by_id["C10"]["forbidden"] is True


def test_forbidden_present_is_penalized_more_than_expected_missing():
    rule = {"weight": 8, "expected": "mandatory"}
    forbidden = compute_rule_score(rule, "forbidden_present")
    missing = compute_rule_score(rule, "missing")
    assert abs(forbidden) > abs(missing)
