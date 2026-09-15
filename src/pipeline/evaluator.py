from __future__ import annotations

import json
import re

from src.core.schema import ConOpsAnalysis, EvaluationReport, MetricResult
from src.llm.prompts import build_evaluator_prompt
from src.llm.providers import BaseLLM
from src.pipeline.quality import (
    classification_correctness,
    get_verdict,
    human_review_stats,
    score_entities,
    score_interfaces,
    score_requirements,
    score_risks,
)
from src.pipeline.semantic_evaluator import semantic_similarity
from src.pipeline.sysml_generator import generate_sysml_v2, validate_sysml_text
from src.domain.rule_based_evaluator import evaluate_document_rules
from src.llm.json_repair import safe_extract_json


WEIGHTS = {
    "completeness": 0.12,
    "strategic_level": 0.10,
    "entities": 0.13,
    "requirements": 0.15,
    "interfaces": 0.15,
    "risks": 0.10,
    "evidence": 0.10,
    "grounding": 0.10,
    "sysml": 0.05,
}

MANDATORY_FIELDS = (
    "purpose_scope", "general_context", "as_is", "to_be", "stakeholders",
    "existing_systems", "existing_services", "expected_services", "gaps",
    "capabilities", "requirements", "interfaces", "risks",
)
OPS_DETAIL_TERMS = (
    "step by step", "button", "click", "procedure", "sequence",
    "detailed scenario", "screen", "cliquer", "bouton",
)
STRATEGIC_TERMS = (
    "vision", "stakeholder", "service", "capability", "gap", "expected",
    "future", "shared", "business", "strategic",
)


def _non_empty(value) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return value is not None


def _text_support_score(items: list[str], source_text: str) -> float:
    if not items:
        return 0.0
    source_tokens = set(re.findall(r"[a-zA-ZÀ-ÿ]{4,}", source_text.lower()))
    supported = 0
    for item in items:
        tokens = set(re.findall(r"[a-zA-ZÀ-ÿ]{4,}", item.lower()))
        if tokens and len(tokens & source_tokens) / len(tokens) >= 0.45:
            supported += 1
    return supported / len(items)


def _traceable_items(analysis: ConOpsAnalysis) -> list:
    return (
        list(analysis.stakeholders)
        + list(analysis.requirements)
        + list(analysis.interfaces)
        + list(analysis.risks)
    )


def _claims(analysis: ConOpsAnalysis) -> list[str]:
    claims = (
        analysis.as_is
        + analysis.to_be
        + analysis.existing_systems
        + analysis.existing_services
        + analysis.expected_services
        + analysis.gaps
        + analysis.capabilities
        + analysis.enablers
        + analysis.constraints
        + analysis.drivers
    )
    claims += [item.name for item in analysis.stakeholders]
    claims += [item.text for item in analysis.requirements]
    claims += [item.exchanged_information for item in analysis.interfaces]
    claims += [item.description for item in analysis.risks]
    return [claim for claim in claims if claim]


def evaluate_rules(
    analysis: ConOpsAnalysis,
    source_text: str,
    reference_text: str = "",
    *,
    rag_used: bool = False,
    evaluator_used: bool = False,
    evaluator_strictness: str = "medium",
) -> tuple[list[MetricResult], float, float, str, float, float]:
    data = analysis.model_dump()
    completeness = (
        sum(_non_empty(data.get(field)) for field in MANDATORY_FIELDS)
        / len(MANDATORY_FIELDS)
        * 100
    )
    source_lower = source_text.lower()
    strategic_hits = sum(source_lower.count(term) for term in STRATEGIC_TERMS)
    operational_hits = sum(source_lower.count(term) for term in OPS_DETAIL_TERMS)
    strategic_score = min(
        100.0,
        100 * (strategic_hits + 2) / (strategic_hits + operational_hits + 4),
    )

    traceable_items = _traceable_items(analysis)
    evidence_texts = [
        evidence.text
        for item in traceable_items
        for evidence in item.evidence
        if evidence.text
    ]
    evidence_coverage = (
        sum(bool(item.evidence) for item in traceable_items) / len(traceable_items)
        if traceable_items
        else 0.0
    )
    evidence_support = _text_support_score(evidence_texts, source_text)
    evidence_score = evidence_coverage * evidence_support * 100

    claim_support = _text_support_score(_claims(analysis), source_text)
    grounding = 0.55 * claim_support + 0.45 * evidence_coverage * evidence_support
    if not rag_used:
        grounding *= 0.90
    if evaluator_used:
        grounding_bonus = {
            "low": 0.04,
            "medium": 0.03,
            "high": 0.02,
        }.get(evaluator_strictness, 0.03)
        grounding = min(1.0, grounding + grounding_bonus * evidence_coverage)

    classification_score = classification_correctness(analysis)
    sysml_text = generate_sysml_v2(analysis)
    sysml_validity, _ = validate_sysml_text(sysml_text, analysis)

    values = {
        "completeness": completeness,
        "strategic_level": strategic_score,
        "entities": score_entities(analysis),
        "requirements": score_requirements(analysis),
        "interfaces": score_interfaces(analysis),
        "risks": score_risks(analysis),
        "evidence": evidence_score,
        "grounding": grounding * 100,
        "sysml": sysml_validity * 100,
    }
    definitions = (
        ("Complétude ConOps", "completeness", "Présence des rubriques clés"),
        ("Niveau stratégique ConOps", "strategic_level", "Vision stratégique vs procédures"),
        ("Qualité des entités", "entities", "Couverture acteurs, services, capacités et systèmes"),
        ("Qualité exigences", "requirements", "Grammaire, preuve, identifiant et concision"),
        ("Interfaces", "interfaces", "Nombre et qualité des échanges traçables"),
        ("Risques", "risks", "Vrais risques avec preuve et mitigation"),
        ("Traçabilité evidence", "evidence", "Couverture et support documentaire des preuves"),
        ("Grounding documentaire", "grounding", "Support lexical des affirmations par la source"),
        ("Validité syntaxique estimée SysML v2", "sysml", "Validation heuristique, non officielle"),
    )
    metrics = [
        MetricResult(
            name=name,
            score=round(values[key], 2),
            weight=WEIGHTS[key],
            comment=comment,
        )
        for name, key, comment in definitions
    ]
    metrics.append(
        MetricResult(
            name="Exactitude de classification",
            score=round(classification_score * 100, 2),
            weight=0.0,
            comment="Métrique diagnostique séparée du grounding documentaire",
        )
    )
    final_score = sum(metric.score * metric.weight for metric in metrics)
    hallucination = 1 - grounding
    hallucination_level = (
        "élevée" if hallucination > 0.55
        else "moyenne" if hallucination > 0.30
        else "faible"
    )
    similarity = (
        semantic_similarity(
            analysis.summary + " " + json.dumps(data, ensure_ascii=False),
            reference_text,
        )
        if reference_text
        else 0.0
    )
    return (
        metrics,
        final_score,
        similarity,
        hallucination_level,
        grounding,
        classification_score,
    )


class ConOpsEvaluator:
    def __init__(
        self,
        llm: BaseLLM | None = None,
        evaluator_strictness: str = "medium",
    ):
        self.llm = llm
        self.evaluator_strictness = (
            evaluator_strictness
            if evaluator_strictness in {"low", "medium", "high"}
            else "medium"
        )

    def evaluate(
        self,
        analysis: ConOpsAnalysis,
        source_text: str,
        configuration: str,
        reference_text: str = "",
        rag_context: str = "",
        knowledge_format: str = "K1",
    ) -> EvaluationReport:
        (
            metrics,
            final_score,
            similarity,
            hallucination_level,
            grounding,
            classification_score,
        ) = evaluate_rules(
            analysis,
            source_text,
            reference_text,
            rag_used=bool(rag_context),
            evaluator_used=self.llm is not None,
            evaluator_strictness=self.evaluator_strictness,
        )
        rule_evaluation = evaluate_document_rules(source_text, analysis)
        if self.llm and rag_context:
            self._apply_evidence_review_policy(analysis)
        analysis.document_classification = rule_evaluation[
            "classification_by_rules"
        ]
        analysis.rule_assessment = rule_evaluation[
            "rule_assessment_detailed"
        ]
        classification = rule_evaluation["classification_by_rules"]
        if classification == "ConOps":
            rule_quality = rule_evaluation["conops_rule_score"]
        elif classification == "OpsCon":
            rule_quality = rule_evaluation["opscon_rule_score"]
        elif classification == "Hybrid":
            rule_quality = (
                rule_evaluation["conops_rule_score"]
                + rule_evaluation["opscon_rule_score"]
            ) / 2
        else:
            rule_quality = max(
                rule_evaluation["conops_rule_score"],
                rule_evaluation["opscon_rule_score"],
            )
        for item in rule_evaluation["rule_assessment_detailed"]:
            item["human_review_needed"] = self._is_relevant_rule_issue(
                item,
                classification,
            )
        rule_weight = {
            "low": 0.60,
            "medium": 0.65,
            "high": 0.70,
        }.get(self.evaluator_strictness, 0.65) if self.llm else 0.70
        final_score = rule_weight * rule_quality + (1.0 - rule_weight) * final_score
        metrics.extend(
            [
                MetricResult(
                    name="ConOps rules V3.0",
                    score=rule_evaluation["conops_rule_score"],
                    weight=0.0,
                    comment="Deterministic score from official ConOps rules.",
                ),
                MetricResult(
                    name="OpsCon rules V3.0",
                    score=rule_evaluation["opscon_rule_score"],
                    weight=0.0,
                    comment="Deterministic score from official OpsCon rules.",
                ),
            ]
        )
        relevant_rule_issues = [
            item
            for item in rule_evaluation["rule_assessment_detailed"]
            if self._is_relevant_rule_issue(item, classification)
        ]
        recommendations = self._recommendations(metrics, analysis)
        recommendations.extend(
            f"{item['rule_id']} - {item['rule_name']}: {item['status']}"
            for item in relevant_rule_issues
        )
        review_count, review_total, _ = human_review_stats(analysis)
        rule_review_count = len(relevant_rule_issues)
        review_count += rule_review_count
        relevant_rule_total = sum(
            self._is_relevant_rule(item, classification)
            for item in rule_evaluation["rule_assessment_detailed"]
        )
        review_total += relevant_rule_total
        review_rate = round(
            review_count / review_total if review_total else 1.0,
            3,
        )
        if self.llm and self.evaluator_strictness in {"low", "medium"}:
            severe_rule_issues = [
                item
                for item in relevant_rule_issues
                if item["status"] == "forbidden_present"
            ]
            if len(severe_rule_issues) < len(relevant_rule_issues):
                review_rate = round(
                    review_rate
                    * (0.75 if self.evaluator_strictness == "low" else 0.90),
                    3,
                )
        evidence_count = sum(
            bool(item["evidence_quote"])
            for item in rule_evaluation["rule_assessment_detailed"]
            if item["status"] in {"present", "forbidden_present"}
        )
        evidence_total = sum(
            item["status"] in {"present", "forbidden_present"}
            for item in rule_evaluation["rule_assessment_detailed"]
        )
        evidence_coverage_score = (
            evidence_count / evidence_total if evidence_total else 0.0
        )
        ontology_fields = (
            analysis.stakeholders,
            analysis.existing_systems,
            analysis.expected_services,
            analysis.capabilities,
            analysis.enablers,
            analysis.requirements,
            analysis.interfaces,
            analysis.risks,
            analysis.gaps,
        )
        ontology_coverage = sum(bool(field) for field in ontology_fields) / len(
            ontology_fields
        )
        rag_usage = (
            _text_support_score(_claims(analysis), rag_context)
            if rag_context
            else 0.0
        )
        if rag_context:
            rag_usage = max(0.01, rag_usage)
        if self.llm:
            try:
                prompt = build_evaluator_prompt(
                    json.dumps(analysis.model_dump(), ensure_ascii=False),
                    source_text,
                    rag_context,
                )
                llm_eval = safe_extract_json(self.llm.generate(prompt, temperature=0.0))
                llm_recommendations = llm_eval.get("recommendations", [])
                if isinstance(llm_recommendations, str):
                    llm_recommendations = [llm_recommendations]
                recommendations.extend(llm_recommendations[:3])
                hallucination_level = str(
                    llm_eval.get("hallucination_level", hallucination_level)
                )
            except Exception:
                pass
        verdict = get_verdict(final_score, review_rate, classification_score)

        return EvaluationReport(
            configuration=configuration,
            metrics=metrics,
            final_score=round(final_score, 2),
            verdict=verdict,
            recommendations=list(dict.fromkeys(recommendations))[:10],
            hallucination_level=hallucination_level,
            semantic_similarity=round(similarity, 3),
            grounding_score=round(grounding, 3),
            classification_correctness_score=round(classification_score, 3),
            concept_coverage_score=round(ontology_coverage, 3),
            ontology_coverage_score=round(ontology_coverage, 3),
            rag_usage_score=round(rag_usage, 3),
            rules_version=rule_evaluation["rules_version"],
            rule_based_classification=classification,
            conops_rule_score=rule_evaluation["conops_rule_score"],
            opscon_rule_score=rule_evaluation["opscon_rule_score"],
            conops_forbidden_penalty=rule_evaluation[
                "conops_forbidden_penalty"
            ],
            opscon_forbidden_penalty=rule_evaluation[
                "opscon_forbidden_penalty"
            ],
            forbidden_penalty=rule_evaluation["forbidden_penalty"],
            missing_mandatory_rules_count=rule_evaluation[
                "missing_mandatory_rules_count"
            ],
            opscon_missing_mandatory_penalty=rule_evaluation[
                "opscon_missing_mandatory_penalty"
            ],
            evidence_coverage_score=round(evidence_coverage_score, 3),
            rule_assessment_detailed=rule_evaluation[
                "rule_assessment_detailed"
            ],
            human_review_count=review_count,
            human_review_rate=review_rate,
            sysml_validity_score=round(
                next(
                    metric.score / 100
                    for metric in metrics
                    if "SysML v2" in metric.name
                ),
                3,
            ),
            sysml_validation_type="heuristic",
        )

    @staticmethod
    def _is_relevant_rule(item: dict, classification: str) -> bool:
        family = item["rule_family"]
        if family == "D":
            return True
        if classification == "ConOps":
            return family == "ConOps"
        if classification == "OpsCon":
            return family == "OpsCon"
        return True

    @classmethod
    def _is_relevant_rule_issue(
        cls,
        item: dict,
        classification: str,
    ) -> bool:
        if not item["human_review_needed"]:
            return False
        if item["rule_family"] != "D":
            return cls._is_relevant_rule(item, classification)
        if classification == "ConOps":
            return (
                item["rule_id"] == "D1" and item["status"] == "missing"
            ) or (
                item["rule_id"] == "D2" and item["status"] == "present"
            )
        if classification == "OpsCon":
            return (
                item["rule_id"] == "D2" and item["status"] == "missing"
            ) or (
                item["rule_id"] == "D1" and item["status"] == "present"
            )
        if classification == "Hybrid":
            return False
        return item["status"] == "missing"

    def _recommendations(
        self,
        metrics: list[MetricResult],
        analysis: ConOpsAnalysis,
    ) -> list[str]:
        recommendations = [
            f"Améliorer : {metric.name} - {metric.comment}"
            for metric in metrics
            if metric.weight > 0 and metric.score < 60
        ]
        if len(analysis.stakeholders) < 6:
            recommendations.append("Valider et compléter la couverture des stakeholders.")
        if len(analysis.interfaces) < 6:
            recommendations.append("Extraire au moins six interfaces métier traçables.")
        recommendations.append(
            "Résultat validé automatiquement, non validé par un expert métier."
        )
        return recommendations

    @staticmethod
    def _apply_evidence_review_policy(analysis: ConOpsAnalysis) -> None:
        for requirement in analysis.requirements:
            if requirement.evidence:
                requirement.reformulation_confidence = max(
                    requirement.reformulation_confidence or 0.0,
                    0.85,
                )
            elif "HUMAN_REVIEW_REQUIRED_NO_EVIDENCE" not in requirement.quality_flags:
                requirement.quality_flags.append("HUMAN_REVIEW_REQUIRED_NO_EVIDENCE")
        for interface in analysis.interfaces:
            if interface.evidence or interface.evidence_quote:
                interface.confidence = max(interface.confidence, 0.85)
            else:
                interface.human_review_needed = True
                if "HUMAN_REVIEW_REQUIRED_NO_EVIDENCE" not in interface.quality_flags:
                    interface.quality_flags.append(
                        "HUMAN_REVIEW_REQUIRED_NO_EVIDENCE"
                    )
