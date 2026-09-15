from __future__ import annotations

from src.domain.rules_loader import build_rules_prompt_context

CONOPS_SCHEMA_INSTRUCTION = """
Result format: one valid JSON object, without Markdown.
Evidence quotes are short and come from the document body. A table of
contents is not evidence unless the same information appears in the body.
Actors, needs, systems, flows and risks are traceable to the source text.
Ambiguous generic environment-wide grouping concepts are excluded.
Generic value entities are excluded. Use stakeholder_needs, expected_outcomes,
value_criteria or value_chain_traceability only when explicit.
System-of-Systems belongs to OpsCon and is not a ConOps concept.

Official scoring is computed deterministically after extraction.
The expected value for "rule_assessment" is [].
Each extracted list remains concise: at most 12 short items per field.

JSON structure:
{
  "document_title": "...",
  "document_classification": "ConOps|OpsCon|Hybrid|Neither",
  "rule_assessment": [],
  "purpose_scope": "...",
  "general_context": "...",
  "current_situation": ["..."],
  "operational_environment": ["..."],
  "change_drivers": ["..."],
  "stakeholder_needs": ["..."],
  "expected_outcomes": ["..."],
  "future_services": ["..."],
  "capability_gaps": ["..."],
  "future_capabilities": ["..."],
  "constraints": ["..."],
  "conops_forbidden_elements": ["..."],
  "opscon_elements": ["..."],
  "operational_flows": ["..."],
  "constituent_systems": ["..."],
  "governance": ["..."],
  "dependencies": ["..."],
  "operational_modes": ["..."],
  "as_is": ["..."],
  "to_be": ["..."],
  "stakeholders": [{"name":"...","role":"...","interest":"...",
    "influence_level":"low|medium|high",
    "evidence":[{"text":"...","source":"document_utilisateur","page":null}]}],
  "existing_systems": ["..."],
  "existing_services": ["..."],
  "expected_services": ["..."],
  "gaps": ["..."],
  "capabilities": ["..."],
  "enablers": ["..."],
  "drivers": ["..."],
  "mitigations": ["..."],
  "requirements": [{"id":"REQ-001","text":"...","priority":"low|medium|high",
    "source_category":"ConOps|OpsCon",
    "evidence":[{"text":"...","source":"document_utilisateur","page":null}]}],
  "interfaces": [{"interface_id":"INT-001","name":"...",
    "source_actor":"...","target_actor":"...","exchanged_item":"...",
    "flow_type":"Information|Material|Energy|Hybrid","direction":"...",
    "trigger":"...","related_rule_id":"O8",
    "criticality":"low|medium|high",
    "evidence":[{"text":"...","source":"document_utilisateur","page":null}]}],
  "risks": [{"description":"...","impact":"low|medium|high",
    "probability":"low|medium|high","mitigation":"...",
    "evidence":[{"text":"...","source":"document_utilisateur","page":null}]}],
  "assumptions": ["..."],
  "future_actions": [{"action":"...","owner":"...","horizon":"...",
    "rationale":"..."}],
  "things_to_change": ["..."],
  "things_to_avoid": ["..."],
  "conops_vs_opscon_comment": "...",
  "summary": "...",
  "recommendations": ["..."]
}
"""

CONOPS_TEMPLATE = """
Separate Problem Space evidence from Solution Space evidence.
For ConOps, focus on current situation, stakeholders, operational environment,
change motivation, needs, expected outcomes, black-box future services,
capability gaps, future capabilities, constraints and shared future vision.
For OpsCon, focus on Future Operational SoS, boundaries, constituent systems,
allocated services and capabilities, executable scenarios, operational flows,
interfaces, governance, dependencies, ownership and operational modes.
"""

CONFIG_PROFILES = {
    "C1": "Baseline: extract directly from the document.",
    "C2": "Use the document structure template.",
    "C3": "Use official ConOps/OpsCon V3.0 rules.",
    "C4": "Use retrieved references as documentary grounding.",
    "C5": "Use official rules, RAG and complementary LLM review.",
    "C6": "Fine-tuned configuration enabled only when a valid adapter is available.",
    "C7": "Fine-tuned configuration with RAG, enabled only when a valid adapter is available.",
}


def build_extraction_prompt(
    text: str,
    *,
    use_template: bool,
    use_rules: bool,
    rag_context: str = "",
    configuration_name: str = "C0",
    knowledge_context: str = "",
) -> str:
    from src.config import settings

    blocks = [
        "Task: systems engineering extraction for ConOps and OpsCon analysis.",
        CONFIG_PROFILES.get(configuration_name, ""),
    ]
    if use_template:
        blocks.append(CONOPS_TEMPLATE)
    if use_rules:
        blocks.extend(
            [
                "Use the official V3.0 rules below as extraction guidance. "
                "Scoring and final classification are computed deterministically.",
                build_rules_prompt_context(),
                knowledge_context,
            ]
        )
    if rag_context:
        blocks.extend(
            [
                "Retrieved reference passages may guide interpretation. "
                "Evidence from the analyzed document remains authoritative:",
                rag_context[:9000],
            ]
        )
    blocks.extend(
        [
            CONOPS_SCHEMA_INSTRUCTION,
            "Document to analyze:",
            text[:settings.llm_document_max_chars],
        ]
    )
    return "\n\n".join(block for block in blocks if block)


def build_evaluator_prompt(
    analysis_json: str,
    source_text: str,
    rag_context: str = "",
) -> str:
    return f"""
Review extraction traceability using the official V3.0 rules.
Deterministic scoring is handled separately. Output JSON with:
{{
  "recommendations": ["..."],
  "hallucination_level": "faible|moyenne|elevee",
  "comment": "..."
}}

RAG context:
{rag_context[:5000]}

Extracted JSON:
{analysis_json[:12000]}

Source:
{source_text[:14000]}
"""
