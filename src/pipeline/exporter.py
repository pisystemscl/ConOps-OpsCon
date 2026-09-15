from __future__ import annotations

from pathlib import Path
import json
import zipfile

import pandas as pd

from src.core.schema import ConOpsAnalysis, EvaluationReport, ExperimentResult
from src.pipeline.quality import (
    classify_risk_candidate,
    requirement_quality_flags,
    text_grounding_similarity,
)
from src.postprocessing.requirement_cleaner import score_requirement_quality
from src.graph.graph_generator import graph_edges_dataframe


def _first_evidence(
    item,
) -> tuple[str, str, int | None, str, str, bool]:
    if not getattr(item, "evidence", None):
        return "", "", None, "", "", False
    evidence = item.evidence[0]
    return (
        evidence.text,
        evidence.source,
        evidence.page,
        evidence.chunk_id,
        evidence.section,
        bool(evidence.text.strip()),
    )


def experiments_summary_dataframe(
    results: list[ExperimentResult],
) -> pd.DataFrame:
    return pd.DataFrame([result.model_dump() for result in results])


def extractions_detailed_dataframe(
    results: list[ExperimentResult],
    analyses: dict[str, ConOpsAnalysis],
) -> pd.DataFrame:
    rows: list[dict] = []
    result_by_code = {result.configuration: result for result in results}

    def base_row(code: str, item_type: str, item_id: str, item_text: str) -> dict:
        result = result_by_code[code]
        return {
            "run_id": result.run_id,
            "configuration": code,
            "document_name": result.document_name,
            "item_type": item_type,
            "item_id": item_id,
            "item_text": item_text,
            "related_rule_id": {
                "stakeholder": "C2",
                "requirement": "C5",
                "interface": "O8",
                "risk": "C13",
                "future_action": "C14",
            }.get(item_type, ""),
            "cleaned_text": item_text,
            "priority": "",
            "source_category": "",
            "source_actor": "",
            "target_actor": "",
            "interface_name": "",
            "exchanged_item": "",
            "flow_type": "",
            "direction": "",
            "trigger": "",
            "owner": "",
            "impact": "",
            "probability": "",
            "mitigation": "",
            "evidence_text": "",
            "evidence_quote": "",
            "evidence_source": "",
            "evidence_page": None,
            "source_page": None,
            "source_document": "",
            "source_chunk_id": "",
            "source_section": "",
            "evidence_start_char": None,
            "evidence_end_char": None,
            "confidence": None,
            "classification_confidence": None,
            "grounding_similarity": 0.0,
            "item_quality_score": 0.0,
            "requirement_quality_score": None,
            "interface_completeness_score": None,
            "validation_status": "automatic_unreviewed",
            "human_review_needed": False,
            "error_flags": "",
            "is_grounded": False,
        }

    def apply_quality(
        row: dict,
        *,
        evidence_text: str,
        evidence_source: str,
        evidence_page: int | None,
        evidence_chunk_id: str = "",
        evidence_section: str = "",
        flags: list[str] | None = None,
        classification_confidence: float | None = None,
    ) -> dict:
        flags = list(flags or [])
        if not evidence_text:
            flags.append("MISSING_EVIDENCE")
        if evidence_page is None:
            flags.append("MISSING_EVIDENCE_PAGE")
        grounding_similarity = text_grounding_similarity(
            row["item_text"],
            evidence_text,
        )
        quality_score = max(
            0.0,
            100.0
            - 18.0 * len(set(flags))
            + 20.0 * grounding_similarity,
        )
        row.update(
            evidence_text=evidence_text,
            evidence_quote=evidence_text,
            evidence_source=evidence_source,
            source_document=evidence_source,
            source_chunk_id=evidence_chunk_id,
            source_section=evidence_section,
            evidence_page=evidence_page,
            source_page=evidence_page,
            classification_confidence=classification_confidence,
            confidence=(
                classification_confidence
                if classification_confidence is not None
                else round(min(1.0, quality_score / 100), 3)
            ),
            grounding_similarity=grounding_similarity,
            item_quality_score=round(min(100.0, quality_score), 2),
            validation_status=(
                "needs_human_review" if flags else "automatic_pass"
            ),
            human_review_needed=bool(flags),
            error_flags="|".join(dict.fromkeys(flags)),
            is_grounded=bool(evidence_text),
        )
        return row

    for code, analysis in analyses.items():
        simple_fields = (
            ("stakeholder_need", analysis.stakeholder_needs, "C5"),
            ("expected_outcome", analysis.expected_outcomes, "C5"),
            ("future_service", analysis.expected_services, "C6"),
            ("capability_gap", analysis.gaps, "C7"),
            ("capability", analysis.capabilities, "C8"),
            ("constituent_system", analysis.constituent_systems, "O6"),
            ("operational_flow", analysis.operational_flows, "O5"),
            ("governance", analysis.governance, "O10"),
            ("dependency", analysis.dependencies, "O11"),
            ("operational_mode", analysis.operational_modes, "O14"),
        )
        for item_type, values, rule_id in simple_fields:
            for index, value in enumerate(values, start=1):
                row = base_row(
                    code,
                    item_type,
                    f"{item_type.upper()}-{index:03d}",
                    value,
                )
                row["related_rule_id"] = rule_id
                apply_quality(
                    row,
                    evidence_text="",
                    evidence_source="",
                    evidence_page=None,
                    flags=["HUMAN_VALIDATION_REQUIRED"],
                )
                rows.append(row)

        for index, stakeholder in enumerate(analysis.stakeholders, start=1):
            row = base_row(code, "stakeholder", f"STK-{index:03d}", stakeholder.name)
            (
                evidence_text,
                evidence_source,
                evidence_page,
                evidence_chunk_id,
                evidence_section,
                grounded,
            ) = _first_evidence(stakeholder)
            row.update(
                source_category=stakeholder.role,
            )
            apply_quality(
                row,
                evidence_text=evidence_text,
                evidence_source=evidence_source,
                evidence_page=evidence_page,
                evidence_chunk_id=evidence_chunk_id,
                evidence_section=evidence_section,
            )
            rows.append(row)

        for requirement in analysis.requirements:
            row = base_row(code, "requirement", requirement.id, requirement.text)
            (
                evidence_text,
                evidence_source,
                evidence_page,
                evidence_chunk_id,
                evidence_section,
                grounded,
            ) = _first_evidence(requirement)
            row.update(
                priority=requirement.priority,
                source_category=requirement.source_category,
            )
            apply_quality(
                row,
                evidence_text=evidence_text,
                evidence_source=evidence_source,
                evidence_page=evidence_page,
                evidence_chunk_id=evidence_chunk_id,
                evidence_section=evidence_section,
                flags=list(
                    dict.fromkeys(
                        requirement_quality_flags(requirement.text)
                        + requirement.quality_flags
                    )
                ),
            )
            row["requirement_quality_score"] = score_requirement_quality(
                requirement
            )
            rows.append(row)

        for index, interface in enumerate(analysis.interfaces, start=1):
            row = base_row(
                code,
                "interface",
                interface.interface_id or f"INT-{index:03d}",
                interface.exchanged_information,
            )
            (
                evidence_text,
                evidence_source,
                evidence_page,
                evidence_chunk_id,
                evidence_section,
                grounded,
            ) = _first_evidence(interface)
            flags = list(interface.quality_flags)
            if not interface.source or not interface.target:
                flags.append("MISSING_INTERFACE_ACTOR")
            if not interface.exchanged_information:
                flags.append("LOW_CONFIDENCE_INTERFACE")
            interface_complete = bool(
                (interface.source_actor or interface.source)
                and (interface.target_actor or interface.target)
                and (interface.exchanged_item or interface.exchanged_information)
                and interface.flow_type
            )
            row.update(
                priority=interface.criticality,
                source_actor=interface.source_actor or interface.source,
                target_actor=interface.target_actor or interface.target,
                interface_name=interface.name,
                exchanged_item=(
                    interface.exchanged_item
                    or interface.exchanged_information
                ),
                flow_type=interface.flow_type,
                direction=interface.direction,
                trigger=interface.trigger,
                related_rule_id=interface.related_rule_id,
                interface_completeness_score=1.0 if interface_complete else 0.0,
            )
            apply_quality(
                row,
                evidence_text=evidence_text,
                evidence_source=evidence_source,
                evidence_page=evidence_page,
                evidence_chunk_id=evidence_chunk_id,
                evidence_section=evidence_section,
                flags=flags,
            )
            rows.append(row)

        for index, risk in enumerate(analysis.risks, start=1):
            row = base_row(code, "risk", f"RISK-{index:03d}", risk.description)
            (
                evidence_text,
                evidence_source,
                evidence_page,
                evidence_chunk_id,
                evidence_section,
                grounded,
            ) = _first_evidence(risk)
            risk_category = classify_risk_candidate(risk.description)
            flags = (
                ["POSSIBLE_WRONG_CATEGORY_RISK"]
                if risk_category != "risk"
                else []
            )
            row.update(
                impact=risk.impact,
                probability=risk.probability,
                mitigation=risk.mitigation,
            )
            apply_quality(
                row,
                evidence_text=evidence_text,
                evidence_source=evidence_source,
                evidence_page=evidence_page,
                evidence_chunk_id=evidence_chunk_id,
                evidence_section=evidence_section,
                flags=flags,
                classification_confidence=1.0 if risk_category == "risk" else 0.35,
            )
            rows.append(row)

        for index, action in enumerate(analysis.future_actions, start=1):
            row = base_row(code, "future_action", f"ACT-{index:03d}", action.action)
            row.update(owner=action.owner, source_category=action.horizon)
            apply_quality(
                row,
                evidence_text="",
                evidence_source="",
                evidence_page=None,
                flags=["HUMAN_VALIDATION_REQUIRED"],
            )
            rows.append(row)

    return pd.DataFrame(rows)


def evaluation_criteria_dataframe(
    results: list[ExperimentResult],
    reports: dict[str, EvaluationReport],
) -> pd.DataFrame:
    rows: list[dict] = []
    result_by_code = {result.configuration: result for result in results}
    for code, report in reports.items():
        result = result_by_code[code]
        recommendations = " | ".join(report.recommendations)
        for metric in report.metrics:
            rows.append(
                {
                    "run_id": result.run_id,
                    "configuration": code,
                    "criterion_name": metric.name,
                    "score": round(metric.score, 3),
                    "weight": metric.weight,
                    "weighted_score": round(metric.score * metric.weight, 3),
                    "comment": metric.comment,
                    "recommendation": recommendations,
                }
            )
    return pd.DataFrame(rows)


def rule_assessment_dataframe(
    results: list[ExperimentResult],
    reports: dict[str, EvaluationReport],
) -> pd.DataFrame:
    rows = []
    result_by_code = {result.configuration: result for result in results}
    for code, report in reports.items():
        result = result_by_code[code]
        for assessment in report.rule_assessment_detailed:
            rows.append(
                {
                    "run_id": result.run_id,
                    "configuration": code,
                    "document_name": result.document_name,
                    "rule_id": assessment["rule_id"],
                    "rule_name": assessment["rule_name"],
                    "rule_family": assessment["rule_family"],
                    "expected": assessment["expected"],
                    "forbidden": assessment["forbidden"],
                    "weight": assessment["weight"],
                    "status": assessment["status"],
                    "matched_terms": assessment.get("matched_terms", ""),
                    "conops_score_contribution": assessment.get(
                        "conops_score_contribution",
                        0.0,
                    ),
                    "opscon_score_contribution": assessment.get(
                        "opscon_score_contribution",
                        0.0,
                    ),
                    "score_contribution": assessment.get(
                        "score_contribution",
                        assessment.get(
                            "conops_score_contribution",
                            assessment.get("opscon_score_contribution", 0.0),
                        ),
                    ),
                    "evidence_quote": assessment["evidence_quote"],
                    "source_page": assessment["source_page"],
                    "comment": assessment["comment"],
                    "human_review_needed": assessment[
                        "human_review_needed"
                    ],
                }
            )
    return pd.DataFrame(rows)


def export_detailed_csv(
    results: list[ExperimentResult],
    analyses: dict[str, ConOpsAnalysis],
    reports: dict[str, EvaluationReport],
    output_dir: str | Path,
    file_prefix: str = "",
    rag_rows: list[dict] | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{file_prefix}_" if file_prefix else ""
    raw_dir = output_dir / "raw_llm_responses"
    failed_dir = output_dir / "failed_json_responses"
    prompts_dir = output_dir / "prompts"
    raw_dir.mkdir(exist_ok=True)
    failed_dir.mkdir(exist_ok=True)
    prompts_dir.mkdir(exist_ok=True)
    for result in results:
        analysis = analyses.get(result.configuration)
        if not analysis:
            failed_response = str(getattr(result, "failed_raw_response", ""))
            if result.status == "failed" and failed_response:
                failed_path = (
                    failed_dir / f"{result.configuration}_failed_response.txt"
                )
                failed_path.write_text(failed_response, encoding="utf-8")
                result.raw_response_path = str(failed_path)
            continue
        raw_response = str(analysis.raw_notes.get("raw_llm_response", ""))
        if raw_response:
            raw_path = raw_dir / f"{result.configuration}_raw_response.txt"
            raw_path.write_text(raw_response, encoding="utf-8")
            result.raw_response_path = str(raw_path)
        if not result.json_valid and raw_response:
            failed_path = failed_dir / f"{result.configuration}_failed_response.txt"
            failed_path.write_text(raw_response, encoding="utf-8")
    paths = {
        "summary": output_dir / f"{prefix}experiments_summary.csv",
        "extractions": output_dir / f"{prefix}extraction_items_detailed.csv",
        "criteria": output_dir / f"{prefix}evaluation_criteria.csv",
        "rag": output_dir / f"{prefix}rag_chunks_used.csv",
        "rules": output_dir / f"{prefix}rule_assessment_detailed.csv",
        "graph_edges": output_dir / f"{prefix}graph_edges.csv",
    }
    frames = {
        "summary": experiments_summary_dataframe(results),
        "extractions": extractions_detailed_dataframe(results, analyses),
        "criteria": evaluation_criteria_dataframe(results, reports),
        "rules": rule_assessment_dataframe(results, reports),
        "graph_edges": pd.concat(
            [
                graph_edges_dataframe(
                    analysis,
                    run_id=next(
                        (
                            item.run_id
                            for item in results
                            if item.configuration == code
                        ),
                        "",
                    ),
                    configuration=code,
                )
                for code, analysis in analyses.items()
            ],
            ignore_index=True,
        )
        if analyses
        else pd.DataFrame(),
        "rag": pd.DataFrame(
            [
                {
                    "run_id": results[0].run_id if results else "",
                    "configuration": row.get("configuration", ""),
                    **row,
                }
                for row in (rag_rows or [])
            ],
            columns=[
                "run_id", "configuration", "query", "chunk_id",
                "rule_id", "source_document", "document_name", "page", "section",
                "embedding_model",
                "retrieval_score", "similarity_score", "distance",
                "used_in_prompt", "used_in_answer", "related_item_ids",
                "related_rule_ids", "chunk_text", "source_path",
            ],
        ),
    }
    for key, dataframe in frames.items():
        dataframe.to_csv(paths[key], index=False, encoding="utf-8-sig")
    return paths


def write_run_metadata(
    run_dir: str | Path,
    results: list[ExperimentResult],
    selected_configurations: list[str],
    *,
    rag_documents: list[str] | None = None,
    limitations: list[str] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    successful = [result for result in results if result.status == "success"]
    best = next(
        (result.configuration for result in successful if result.best_configuration),
        None,
    )
    first = results[0] if results else None
    metadata = {
        "run_id": first.run_id if first else "",
        "timestamp": first.timestamp if first else "",
        "document_name": first.document_name if first else "",
        "document_hash": first.document_hash if first else "",
        "selected_configurations": selected_configurations,
        "experiment_type": first.experiment_type if first else "",
        "fixed_variable": first.fixed_variable if first else "",
        "variable_tested": first.variable_tested if first else "",
        "knowledge_format": first.knowledge_format if first else "",
        "rules_version": first.rules_version if first else "3.0",
        "rule_based_classification": (
            first.rule_based_classification if first else "Neither"
        ),
        "repeat_count": first.repeat_count if first else 0,
        "ollama_models": sorted(
            {result.model_name for result in results if result.model_name}
        ),
        "rag_documents": rag_documents or [],
        "best_configuration": best,
        "configurations": [
            {
                "code": result.configuration,
                "status": result.status,
                "model_name": result.model_name,
                "config_hash": result.config_hash,
                "prompt_hash": result.prompt_hash,
                "repeat_count": result.repeat_count,
                "mean_score": result.mean_score,
                "std_score": result.std_score,
            }
            for result in results
        ],
        "validation_notice": (
            "Resultats valides automatiquement, non valides par un expert metier."
        ),
        "sysml_validation_type": "heuristic",
        "limitations": limitations or [
            "Evaluation automatique a confirmer par un expert metier.",
            "Validation SysML heuristique et non officielle.",
        ],
    }
    path = run_dir / "run_metadata.json"
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def zip_run_directory(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    zip_path = run_dir / f"{run_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in run_dir.rglob("*"):
            if path.is_file() and path != zip_path:
                archive.write(path, path.relative_to(run_dir))
    return zip_path
