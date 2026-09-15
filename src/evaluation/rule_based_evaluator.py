from __future__ import annotations

from src.domain import rule_based_evaluator as domain_evaluator
from src.rules.rule_repository import RulesRepository, load_official_rules_repository


def evaluate_document_rules(analysis, rules_repository: RulesRepository | None = None) -> dict:
    repository = rules_repository or load_official_rules_repository()
    source_text = str(getattr(analysis, "raw_notes", {}).get("source_text", ""))
    return domain_evaluator.evaluate_document_rules(
        source_text,
        analysis=analysis,
        rules=repository.data,
    )


def evaluate_discriminating_rules(analysis) -> list[dict]:
    result = evaluate_document_rules(analysis)
    return [
        row for row in result["rule_assessment_detailed"]
        if row["rule_family"] == "D"
    ]


def evaluate_conops_rules(analysis) -> list[dict]:
    result = evaluate_document_rules(analysis)
    return [
        row for row in result["rule_assessment_detailed"]
        if row["rule_family"] == "ConOps"
    ]


def evaluate_opscon_rules(analysis) -> list[dict]:
    result = evaluate_document_rules(analysis)
    return [
        row for row in result["rule_assessment_detailed"]
        if row["rule_family"] == "OpsCon"
    ]


def compute_rule_contribution(rule: dict, status: str) -> float:
    return domain_evaluator.compute_rule_score(rule, status)


def classify_document_from_rules(rule_results: dict) -> str:
    return domain_evaluator.classify_document(
        float(rule_results.get("conops_rule_score", 0.0)),
        float(rule_results.get("opscon_rule_score", 0.0)),
    )


def compute_rule_confidence(rule_results: dict) -> float:
    rows = rule_results.get("rule_assessment_detailed", [])
    if not rows:
        return 0.0
    supported = sum(bool(row.get("evidence_quote")) for row in rows)
    clean = sum(row.get("status") != "forbidden_present" for row in rows)
    return round((supported / len(rows) * 0.6) + (clean / len(rows) * 0.4), 3)


def compute_forbidden_penalties(rule_results: dict) -> float:
    return float(rule_results.get("forbidden_penalty", 0.0))


def compute_missing_mandatory_rules(rule_results: dict) -> list[dict]:
    return list(rule_results.get("missing_mandatory_rules", []))


def generate_rule_explanation(rule_results: dict) -> str:
    classification = classify_document_from_rules(rule_results)
    missing = compute_missing_mandatory_rules(rule_results)
    forbidden = rule_results.get("forbidden_violations", [])
    return (
        f"Classification deterministe: {classification}. "
        f"Score ConOps={rule_results.get('conops_rule_score', 0.0)}/100; "
        f"Score OpsCon={rule_results.get('opscon_rule_score', 0.0)}/100. "
        f"Regles obligatoires manquantes={len(missing)}; "
        f"violations interdites={len(forbidden)}."
    )
