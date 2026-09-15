from __future__ import annotations
import json
from typing import Annotated, Any
from pydantic import BaseModel, BeforeValidator, Field, field_validator, model_validator


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("description", "text", "name", "action", "title", "value", "content"):
            candidate = value.get(key)
            if candidate is not None:
                return _coerce_text(candidate)
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()

class Evidence(BaseModel):
    text: str = ""
    source: str = "document_utilisateur"
    page: int | None = None
    chunk_id: str = ""
    section: str = ""
    grounding_similarity: float | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_evidence(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"text": value}
        return value


def _normalize_evidence_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


EvidenceList = Annotated[list[Evidence], BeforeValidator(_normalize_evidence_list)]

class Stakeholder(BaseModel):
    name: str
    role: str = ""
    interest: str = ""
    influence_level: str = "medium"
    evidence: EvidenceList = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_stakeholder(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"name": value}
        if isinstance(value, dict):
            normalized = dict(value)
            if "name" not in normalized:
                name = (
                    normalized.get("stakeholder")
                    or normalized.get("actor")
                    or normalized.get("role")
                )
                if name:
                    normalized["name"] = _coerce_text(name)
            for field_name in ("name", "role", "interest", "influence_level"):
                if field_name in normalized:
                    normalized[field_name] = _coerce_text(normalized[field_name])
            return normalized
        return value

class Requirement(BaseModel):
    id: str
    text: str
    priority: str = "medium"
    source_category: str = "ConOps"
    evidence: EvidenceList = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    reformulation_confidence: float | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_requirement(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"id": "", "text": value}
        if isinstance(value, dict):
            text = (
                value.get("text")
                or value.get("requirement")
                or value.get("description")
            )
            if text is not None and "text" not in value:
                value = {**value, "text": _coerce_text(text)}
            if "text" in value and "id" not in value:
                value = {**value, "id": ""}
        return value

    @model_validator(mode="after")
    def fill_requirement_id(self):
        if not self.id:
            self.id = "REQ-UNNUMBERED"
        return self

class Interface(BaseModel):
    source: str = ""
    target: str = ""
    exchanged_information: str = ""
    criticality: str = "medium"
    evidence: EvidenceList = Field(default_factory=list)
    interface_id: str = ""
    name: str = ""
    source_actor: str = ""
    target_actor: str = ""
    exchanged_item: str = ""
    flow_type: str = "Information"
    direction: str = ""
    trigger: str = ""
    related_rule_id: str = "O8"
    evidence_quote: str = ""
    source_page: int | None = None
    confidence: float = 0.0
    human_review_needed: bool = False
    quality_flags: list[str] = Field(default_factory=list)

    @field_validator("flow_type")
    @classmethod
    def validate_flow_type(cls, value: str) -> str:
        allowed = {"Information", "Material", "Energy", "Hybrid"}
        return value if value in allowed else "Information"

    @model_validator(mode="before")
    @classmethod
    def normalize_interface(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"exchanged_information": value}
        if isinstance(value, dict):
            exchanged = (
                value.get("exchanged_information")
                or value.get("exchanged_item")
                or value.get("information")
                or value.get("description")
            )
            if exchanged is not None and "exchanged_information" not in value:
                return {**value, "exchanged_information": _coerce_text(exchanged)}
        return value

    @model_validator(mode="after")
    def synchronize_interface_fields(self):
        self.source_actor = self.source_actor or self.source
        self.target_actor = self.target_actor or self.target
        self.exchanged_item = self.exchanged_item or self.exchanged_information
        self.source = self.source or self.source_actor
        self.target = self.target or self.target_actor
        self.exchanged_information = (
            self.exchanged_information or self.exchanged_item
        )
        self.name = self.name or self.exchanged_item
        if self.evidence and not self.evidence_quote:
            self.evidence_quote = self.evidence[0].text
        if self.evidence and self.source_page is None:
            self.source_page = self.evidence[0].page
        self.confidence = self.confidence or (
            1.0
            if self.source_actor and self.target_actor and self.exchanged_item
            else 0.45
        )
        if not (self.source_actor and self.target_actor and self.exchanged_item):
            self.human_review_needed = True
            if "MISSING_INTERFACE_FIELD" not in self.quality_flags:
                self.quality_flags.append("MISSING_INTERFACE_FIELD")
        self.direction = self.direction or (
            f"{self.source_actor} -> {self.target_actor}"
            if self.source_actor and self.target_actor
            else ""
        )
        return self

class Risk(BaseModel):
    description: str
    impact: str = "medium"
    probability: str = "medium"
    mitigation: str = ""
    evidence: EvidenceList = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_risk(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"description": value}
        if isinstance(value, dict) and "description" not in value:
            risk = value.get("risk") or value.get("text") or value.get("name")
            if risk:
                return {**value, "description": _coerce_text(risk)}
        return value

    @field_validator("mitigation", mode="before")
    @classmethod
    def normalize_mitigation(cls, value: Any) -> str:
        if isinstance(value, list):
            return "; ".join(text for item in value if (text := _coerce_text(item)))
        return _coerce_text(value)

class ActionItem(BaseModel):
    action: str
    owner: str = "à définir"
    horizon: str = "future"
    rationale: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_action(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"action": value}
        return value

class ConOpsAnalysis(BaseModel):
    document_title: str = "ConOps analysé"
    document_classification: str = "Neither"
    rule_assessment: list[dict[str, Any]] = Field(default_factory=list)
    purpose_scope: str = ""
    general_context: str = ""
    current_situation: list[str] = Field(default_factory=list)
    operational_environment: list[str] = Field(default_factory=list)
    change_drivers: list[str] = Field(default_factory=list)
    stakeholder_needs: list[str] = Field(default_factory=list)
    expected_outcomes: list[str] = Field(default_factory=list)
    future_services: list[str] = Field(default_factory=list)
    capability_gaps: list[str] = Field(default_factory=list)
    future_capabilities: list[str] = Field(default_factory=list)
    conops_forbidden_elements: list[str] = Field(default_factory=list)
    opscon_elements: list[str] = Field(default_factory=list)
    operational_flows: list[str] = Field(default_factory=list)
    constituent_systems: list[str] = Field(default_factory=list)
    governance: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    operational_modes: list[str] = Field(default_factory=list)
    as_is: list[str] = Field(default_factory=list)
    to_be: list[str] = Field(default_factory=list)
    stakeholders: list[Stakeholder] = Field(default_factory=list)
    existing_systems: list[str] = Field(default_factory=list)
    existing_services: list[str] = Field(default_factory=list)
    expected_services: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    enablers: list[str] = Field(default_factory=list)
    drivers: list[str] = Field(default_factory=list)
    mitigations: list[str] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    interfaces: list[Interface] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    future_actions: list[ActionItem] = Field(default_factory=list)
    things_to_change: list[str] = Field(default_factory=list)
    things_to_avoid: list[str] = Field(default_factory=list)
    conops_vs_opscon_comment: str = ""
    summary: str = ""
    raw_notes: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "as_is",
        "to_be",
        "current_situation",
        "operational_environment",
        "change_drivers",
        "stakeholder_needs",
        "expected_outcomes",
        "future_services",
        "capability_gaps",
        "future_capabilities",
        "conops_forbidden_elements",
        "opscon_elements",
        "operational_flows",
        "constituent_systems",
        "governance",
        "dependencies",
        "operational_modes",
        "existing_systems",
        "existing_services",
        "expected_services",
        "gaps",
        "capabilities",
        "enablers",
        "drivers",
        "mitigations",
        "constraints",
        "assumptions",
        "things_to_change",
        "things_to_avoid",
        mode="before",
    )
    @classmethod
    def normalize_text_lists(cls, value: Any) -> list[str]:
        if value is None:
            return []
        values = value if isinstance(value, list) else [value]
        return [text for item in values if (text := _coerce_text(item))]

class MetricResult(BaseModel):
    name: str
    score: float
    weight: float
    comment: str

class EvaluationReport(BaseModel):
    configuration: str
    metrics: list[MetricResult]
    final_score: float
    verdict: str
    recommendations: list[str]
    hallucination_level: str
    semantic_similarity: float
    grounding_score: float
    classification_correctness_score: float = 0.0
    concept_coverage_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    ontology_coverage_score: float = 0.0
    rag_usage_score: float = 0.0
    expert_agreement_score: float | None = None
    rules_version: str = "3.0"
    rule_based_classification: str = "Neither"
    conops_rule_score: float = 0.0
    opscon_rule_score: float = 0.0
    conops_forbidden_penalty: float = 0.0
    opscon_forbidden_penalty: float = 0.0
    forbidden_penalty: float = 0.0
    missing_mandatory_rules_count: int = 0
    opscon_missing_mandatory_penalty: float = 0.0
    evidence_coverage_score: float = 0.0
    rule_assessment_detailed: list[dict[str, Any]] = Field(default_factory=list)
    human_review_count: int = 0
    human_review_rate: float = 0.0
    sysml_validity_score: float
    sysml_validation_type: str = "heuristic"

    @field_validator("verdict", mode="before")
    @classmethod
    def normalize_verdict(cls, value: Any) -> str:
        if isinstance(value, dict):
            return "; ".join(
                f"{_coerce_text(key)}: {_coerce_text(item)}"
                for key, item in value.items()
            )
        if isinstance(value, list):
            return "; ".join(
                text for item in value if (text := _coerce_text(item))
            )
        return _coerce_text(value)

class ExperimentResult(BaseModel):
    run_id: str
    timestamp: str
    document_name: str
    document_hash: str = ""
    prompt_hash: str = ""
    config_hash: str = ""
    configuration: str
    description: str
    experiment_type: str = "strategy_comparison"
    fixed_variable: str = ""
    variable_tested: str = "strategy"
    strategy_name: str = ""
    knowledge_format: str = "K1"
    provider: str
    model_name: str
    rag_used: bool
    rules_used: bool
    evaluator_used: bool
    llm_evaluator_used: bool = False
    fine_tuned_used: bool
    status: str = "success"
    error_message: str = ""
    execution_time_sec: float = 0.0
    repeat_count: int = 1
    mean_score: float | None = None
    std_score: float | None = None
    stability_score: float | None = None
    final_score: float | None = None
    conops_score: float | None = None
    sysml_score: float | None = None
    semantic_similarity: float | None = None
    hallucination_level: str = ""
    hallucination_score: float | None = None
    grounding_score: float | None = None
    classification_correctness_score: float | None = None
    concept_coverage_score: float | None = None
    ontology_coverage_score: float | None = None
    rag_usage_score: float | None = None
    rules_version: str = "3.0"
    rules_integrity_valid: bool = True
    classification_rule_based: str = "Neither"
    rule_based_classification: str = "Neither"
    conops_rule_score: float | None = None
    opscon_rule_score: float | None = None
    hybrid_score: float | None = None
    neither_score: float | None = None
    rule_compliance_score: float | None = None
    conops_forbidden_penalty: float | None = None
    opscon_forbidden_penalty: float | None = None
    forbidden_penalty_total: float | None = None
    forbidden_penalty: float | None = None
    mandatory_missing_count: int = 0
    recommended_missing_count: int = 0
    optional_present_count: int = 0
    missing_mandatory_rules_count: int = 0
    opscon_missing_mandatory_penalty: float | None = None
    evidence_coverage_score: float | None = None
    rule_confidence_score: float | None = None
    rag_traceability_score: float | None = None
    rule_evidence_coverage_score: float | None = None
    json_validity_rate: float | None = None
    schema_compliance_rate: float | None = None
    json_valid: bool = True
    json_repair_attempts: int = 0
    json_error_message: str = ""
    raw_response_path: str = ""
    failed_raw_response: str = Field(default="", exclude=True)
    retrieved_chunks_count: int = 0
    mean_retrieval_score: float | None = None
    top_chunk_source: str = ""
    top_chunk_page: int | None = None
    rag_context_tokens: int = 0
    retrieval_hit_rate: float | None = None
    chunks_used_in_answer: int = 0
    evidence_from_rag_count: int = 0
    repeat_index: int = 0
    temperature: float = 0.0
    opscon_score: float | None = None
    sysml_validation_type: str = "heuristic"
    requirements_count: int = 0
    stakeholders_count: int = 0
    risks_count: int = 0
    interfaces_count: int = 0
    future_actions_count: int = 0
    evidence_count: int = 0
    sysml_elements_count: int = 0
    human_review_count: int = 0
    human_review_rate: float = 0.0
    best_configuration: bool = False
    execution_notes: str = ""
