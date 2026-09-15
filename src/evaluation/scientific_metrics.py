from __future__ import annotations


def compute_scientific_metrics(
    rule_results: dict,
    *,
    grounding_score: float = 0.0,
    stability_score: float = 0.0,
    human_review_rate: float = 0.0,
    json_valid: bool = True,
    schema_compliant: bool = True,
    rag_traceability_score: float = 0.0,
) -> dict:
    rows = rule_results.get("rule_assessment_detailed", [])
    total = len(rows) or 1
    forbidden_present = sum(
        row.get("status") == "forbidden_present" for row in rows
    )
    mandatory_missing = sum(
        row.get("expected") == "mandatory" and row.get("status") == "missing"
        for row in rows
    )
    present_with_evidence = sum(
        row.get("status") in {"present", "forbidden_present"}
        and bool(row.get("evidence_quote"))
        for row in rows
    )
    present_total = (
        sum(row.get("status") in {"present", "forbidden_present"} for row in rows)
        or 1
    )
    conops_score = float(rule_results.get("conops_rule_score", 0.0))
    opscon_score = float(rule_results.get("opscon_rule_score", 0.0))
    return {
        "rule_compliance_score": max(conops_score, opscon_score),
        "evidence_coverage_score": round(present_with_evidence / present_total, 3),
        "grounding_score": grounding_score,
        "hallucination_risk_score": round(1.0 - grounding_score, 3),
        "json_validity_rate": 1.0 if json_valid else 0.0,
        "schema_compliance_rate": 1.0 if schema_compliant else 0.0,
        "stability_score": stability_score,
        "extraction_consistency_score": stability_score,
        "rag_traceability_score": rag_traceability_score,
        "human_review_rate": human_review_rate,
        "forbidden_violation_rate": round(forbidden_present / total, 3),
        "mandatory_missing_rate": round(mandatory_missing / total, 3),
    }
