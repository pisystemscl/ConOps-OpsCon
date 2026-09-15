from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from src.domain.rules_loader import all_rules, load_rules
from src.pipeline.quality import split_source_pages


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text))
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", value.lower()).strip()


def _contains(text: str, terms: Iterable[str]) -> list[str]:
    normalized = _normalize(text)
    return [term for term in terms if _normalize(term) in normalized]


def _sentences(source_text: str) -> list[tuple[int | None, str]]:
    pages = split_source_pages(source_text)
    sources = pages or [(None, source_text)]
    output = []
    for page, text in sources:
        for sentence in re.split(r"(?<=[.!?;:])\s+|\n+", text):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if len(sentence) >= 20:
                output.append((page, sentence))
    return output


def _evidence_for_terms(
    source_text: str,
    terms: Iterable[str],
) -> tuple[str, int | None]:
    normalized_terms = [_normalize(term) for term in terms if term]
    for page, sentence in _sentences(source_text):
        normalized_sentence = _normalize(sentence)
        if any(term in normalized_sentence for term in normalized_terms):
            return sentence[:500], page
    return "", None


def _structured_hints(analysis) -> dict[str, bool]:
    if analysis is None:
        return {}
    return {
        "C1": bool(analysis.as_is or analysis.existing_services),
        "C2": bool(analysis.stakeholders),
        "C4": bool(analysis.drivers or analysis.gaps),
        "C5": bool(analysis.raw_notes.get("stakeholder_needs")),
        "C6": bool(analysis.expected_services),
        "C7": bool(analysis.gaps),
        "C8": bool(analysis.capabilities),
        "C12": bool(analysis.to_be),
        "C13": bool(analysis.constraints),
        "O3": bool(analysis.raw_notes.get("service_allocations")),
        "O5": bool(analysis.raw_notes.get("operational_flows")),
        "O6": bool(analysis.raw_notes.get("constituent_systems")),
        "O8": bool(analysis.interfaces),
        "O10": bool(analysis.raw_notes.get("governance")),
        "O11": bool(analysis.raw_notes.get("dependencies")),
        "O14": bool(analysis.raw_notes.get("operational_modes")),
    }


def assess_rules(source_text: str, analysis=None, rules: dict | None = None) -> list[dict]:
    rules = rules or load_rules()
    hints = _structured_hints(analysis)
    assessments = []
    for rule in all_rules(rules):
        evidence_terms = rule.get("evidence_keywords", [])
        forbidden_terms = rule.get("forbidden_elements", [])
        evidence_hits = _contains(source_text, evidence_terms)
        forbidden_hits = _contains(source_text, forbidden_terms)
        detected = bool(evidence_hits or hints.get(rule["id"]))
        is_forbidden_rule = bool(rule.get("forbidden"))

        if is_forbidden_rule:
            status = "forbidden_present" if forbidden_hits else "forbidden_absent"
            matched_terms = forbidden_hits
        elif forbidden_hits:
            status = "forbidden_present"
            matched_terms = forbidden_hits
        elif detected:
            status = "present"
            matched_terms = evidence_hits
        elif rule.get("expected") == "optional":
            status = "not_applicable"
            matched_terms = []
        else:
            status = "missing"
            matched_terms = []

        evidence_quote, source_page = _evidence_for_terms(
            source_text,
            matched_terms or evidence_hits,
        )
        expected = rule.get("expected")
        if expected is None:
            expected_for = rule.get("expected_for", [])
            forbidden_for = rule.get("forbidden_for", [])
            expected = (
                f"expected_for={','.join(expected_for) or 'none'}; "
                f"forbidden_for={','.join(forbidden_for) or 'none'}"
            )
        assessments.append(
            {
                "rule_id": rule["id"],
                "rule_name": rule["name"],
                "rule_family": rule["family"],
                "expected": expected,
                "forbidden": is_forbidden_rule,
                "weight": float(rule["weight"]),
                "status": status,
                "matched_terms": ", ".join(matched_terms),
                "evidence_quote": evidence_quote,
                "source_page": source_page,
                "comment": rule["description"],
                "human_review_needed": status in {
                    "missing",
                    "forbidden_present",
                },
            }
        )
    return assessments


def _contribution(
    assessment: dict,
    expected: str,
    *,
    alpha: float,
    beta: float,
    recommended_factor: float,
) -> float:
    weight = assessment["weight"]
    status = assessment["status"]
    if status == "forbidden_present":
        return -beta * weight
    if status in {"present", "forbidden_absent"}:
        return weight
    if expected == "mandatory":
        return -alpha * weight
    if expected == "recommended":
        return -recommended_factor * weight
    return 0.0


def compute_rule_score(
    rule: dict,
    status: str,
    rules: dict | None = None,
) -> float:
    """Compute one deterministic rule contribution from the V3.0 policy."""
    rules = rules or load_rules()
    normalized_status = {
        "expected_present": "present",
        "expected_and_present": "present",
        "expected_missing": "missing",
        "expected_and_missing": "missing",
        "forbidden_and_present": "forbidden_present",
        "forbidden_and_absent": "forbidden_absent",
    }.get(_normalize(status).replace(" ", "_"), status)
    policy = rules["scoring_policy"]
    assessment = {
        "weight": float(rule["weight"]),
        "status": normalized_status,
    }
    return round(
        _contribution(
            assessment,
            rule.get("expected", "mandatory"),
            alpha=float(policy["alpha"]),
            beta=float(policy["beta"]),
            recommended_factor=float(
                policy.get("recommended_missing_factor", policy["alpha"])
            ),
        ),
        3,
    )


def _normalize_score(raw_score: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return round(max(0.0, min(100.0, raw_score / maximum * 100)), 2)


def classify_document(
    conops_score: float,
    opscon_score: float,
    rules: dict | None = None,
) -> str:
    rules = rules or load_rules()
    thresholds = rules["classification_policy"]
    high = float(thresholds["high_score_threshold"])
    minimum = float(thresholds["minimum_score_threshold"])
    if conops_score >= high and opscon_score >= high:
        return "Hybrid"
    if conops_score >= high and conops_score > opscon_score:
        return "ConOps"
    if opscon_score >= high and opscon_score > conops_score:
        return "OpsCon"
    if conops_score < minimum and opscon_score < minimum:
        return "Neither"
    return "ConOps" if conops_score >= opscon_score else "OpsCon"


def score_assessments(
    assessments: list[dict],
    rules: dict | None = None,
) -> dict:
    rules = rules or load_rules()
    policy = rules["scoring_policy"]
    alpha = float(policy["alpha"])
    beta = float(policy["beta"])
    recommended_factor = float(policy["recommended_missing_factor"])
    by_id = {assessment["rule_id"]: assessment for assessment in assessments}

    conops_raw = 0.0
    opscon_raw = 0.0
    conops_max = 0.0
    opscon_max = 0.0
    conops_forbidden_penalty = 0.0
    opscon_forbidden_penalty = 0.0
    opscon_missing_mandatory_penalty = 0.0

    for rule in rules["fundamental_rules"]:
        assessment = by_id[rule["id"]]
        weight = float(rule["weight"])
        if "ConOps" in rule["expected_for"]:
            conops_max += weight
            contribution = weight if assessment["status"] == "present" else -alpha * weight
            conops_raw += contribution
            assessment["conops_score_contribution"] = round(contribution, 3)
        if "ConOps" in rule["forbidden_for"] and assessment["status"] == "present":
            contribution = -beta * weight
            conops_raw += contribution
            conops_forbidden_penalty += abs(contribution)
            assessment["conops_score_contribution"] = round(contribution, 3)
        if "OpsCon" in rule["expected_for"]:
            opscon_max += weight
            contribution = weight if assessment["status"] == "present" else -alpha * weight
            opscon_raw += contribution
            assessment["opscon_score_contribution"] = round(contribution, 3)
        if "OpsCon" in rule["forbidden_for"] and assessment["status"] == "present":
            contribution = -beta * weight
            opscon_raw += contribution
            opscon_forbidden_penalty += abs(contribution)
            assessment["opscon_score_contribution"] = round(contribution, 3)

    for rule in rules["conops_rules"]:
        assessment = by_id[rule["id"]]
        conops_max += float(rule["weight"])
        contribution = _contribution(
            assessment,
            rule["expected"],
            alpha=alpha,
            beta=beta,
            recommended_factor=recommended_factor,
        )
        if assessment["status"] == "forbidden_present":
            conops_forbidden_penalty += abs(contribution)
        conops_raw += contribution
        assessment["score_contribution"] = round(contribution, 3)
        assessment["conops_score_contribution"] = round(contribution, 3)

    for rule in rules["opscon_rules"]:
        assessment = by_id[rule["id"]]
        opscon_max += float(rule["weight"])
        contribution = _contribution(
            assessment,
            rule["expected"],
            alpha=alpha,
            beta=beta,
            recommended_factor=recommended_factor,
        )
        if rule["expected"] == "mandatory" and assessment["status"] == "missing":
            opscon_missing_mandatory_penalty += abs(contribution)
        opscon_raw += contribution
        assessment["score_contribution"] = round(contribution, 3)
        assessment["opscon_score_contribution"] = round(contribution, 3)

    conops_score = _normalize_score(conops_raw, conops_max)
    opscon_score = _normalize_score(opscon_raw, opscon_max)
    classification = classify_document(conops_score, opscon_score, rules)
    forbidden_penalty = (
        conops_forbidden_penalty + opscon_forbidden_penalty
    )
    missing_mandatory_rules = [
        item
        for item in assessments
        if item["status"] == "missing" and item["expected"] == "mandatory"
    ]

    return {
        "rules_version": rules["version"],
        "classification_by_rules": classification,
        "conops_rule_score": conops_score,
        "opscon_rule_score": opscon_score,
        "conops_forbidden_penalty": round(conops_forbidden_penalty, 3),
        "opscon_forbidden_penalty": round(opscon_forbidden_penalty, 3),
        "forbidden_penalty": round(forbidden_penalty, 3),
        "opscon_missing_mandatory_penalty": round(
            opscon_missing_mandatory_penalty,
            3,
        ),
        "missing_mandatory_rules_count": len(missing_mandatory_rules),
        "rule_assessment_detailed": assessments,
        "forbidden_violations": [
            item for item in assessments if item["status"] == "forbidden_present"
        ],
        "missing_mandatory_rules": missing_mandatory_rules,
    }


def evaluate_document_rules(
    source_text: str,
    analysis=None,
    rules: dict | None = None,
) -> dict:
    rules = rules or load_rules()
    return score_assessments(
        assess_rules(source_text, analysis=analysis, rules=rules),
        rules=rules,
    )


def evaluate_rules(analysis_json) -> dict:
    """Evaluate parsed LLM output while keeping scoring authority in Python."""
    if hasattr(analysis_json, "model_dump"):
        analysis = analysis_json
        source_text = str(analysis.raw_notes.get("source_text", ""))
    elif isinstance(analysis_json, dict):
        from src.core.schema import ConOpsAnalysis

        source_text = str(
            analysis_json.get("source_text")
            or analysis_json.get("document_text")
            or ""
        )
        analysis = ConOpsAnalysis.model_validate(analysis_json)
    else:
        raise TypeError("analysis_json must be a mapping or ConOpsAnalysis")
    return evaluate_document_rules(source_text, analysis=analysis)


def related_rule_ids_for_text(
    text: str,
    rules: dict | None = None,
) -> list[str]:
    rules = rules or load_rules()
    related = []
    for rule in all_rules(rules):
        terms = [
            *rule.get("evidence_keywords", []),
            *rule.get("forbidden_elements", []),
        ]
        if _contains(text, terms):
            related.append(rule["id"])
    return related
