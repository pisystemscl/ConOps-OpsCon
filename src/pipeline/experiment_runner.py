from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean, pstdev
from time import perf_counter
from uuid import uuid4

import pandas as pd

from src.config import settings
from src.core.rag_engine import RagEngine
from src.core.schema import ConOpsAnalysis, EvaluationReport, ExperimentResult
from src.llm.providers import BaseLLM, get_llm_for_configuration
from src.llm.fine_tuned_provider import fine_tuning_readiness
from src.pipeline.evaluator import ConOpsEvaluator
from src.pipeline.extractor import ConOpsExtractor
from src.pipeline.quality import get_verdict
from src.pipeline.sysml_generator import generate_sysml_v2, validate_sysml_text


@dataclass(frozen=True)
class Configuration:
    code: str
    description: str
    use_template: bool
    use_rules: bool
    use_rag: bool
    use_llm_evaluator: bool
    uses_finetuned_model: bool = False


CONFIGURATIONS = [
    Configuration("C1", "LLM seul", False, False, False, False),
    Configuration("C2", "LLM + template ConOps", True, False, False, False),
    Configuration("C3", "LLM + regles ConOps/OpsCon", True, True, False, False),
    Configuration("C4", "LLM + RAG", True, False, True, False),
    Configuration("C5", "LLM + RAG + evaluateur", True, True, True, True),
    Configuration("C6", "Modele fine-tune", True, True, False, False, True),
    Configuration("C7", "Modele fine-tune + RAG", True, True, True, True, True),
]


def _hash_payload(payload) -> str:
    serialized = (
        payload if isinstance(payload, str)
        else json.dumps(payload, sort_keys=True, ensure_ascii=False)
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ExperimentRunner:
    def __init__(
        self,
        llm: BaseLLM | None = None,
        rag: RagEngine | None = None,
        reference_text: str = "",
        provider: str | None = None,
        models_by_configuration: dict[str, str] | None = None,
        knowledge_format: str = "K1",
        evaluator_strictness: str = "medium",
    ):
        self.llm = llm
        self.rag = rag
        self.reference_text = reference_text
        self.provider = provider
        self.models_by_configuration = models_by_configuration or {}
        self.knowledge_format = knowledge_format
        self.evaluator_strictness = evaluator_strictness
        self.errors: dict[str, str] = {}
        self.run_id = ""
        self.timestamp = ""
        self.document_hash = ""
        self.rag_rows: list[dict] = []

    @staticmethod
    def _evidence_count(analysis: ConOpsAnalysis) -> int:
        items = (
            list(analysis.stakeholders)
            + list(analysis.requirements)
            + list(analysis.interfaces)
            + list(analysis.risks)
        )
        return sum(len(item.evidence) for item in items)

    @staticmethod
    def _sysml_element_count(analysis: ConOpsAnalysis) -> int:
        return (
            1
            + len(analysis.stakeholders)
            + len(analysis.existing_systems)
            + len(analysis.expected_services)
            + len(analysis.capabilities)
            + len(analysis.enablers)
            + len(analysis.requirements)
            + len(analysis.interfaces)
        )

    def _llm_for(self, configuration: Configuration) -> BaseLLM:
        model = (
            settings.finetuned_model_path
            if configuration.uses_finetuned_model
            else self.models_by_configuration.get(configuration.code)
        )
        try:
            return get_llm_for_configuration(
                configuration.code,
                provider=self.provider,
                model=model,
            )
        except Exception:
            if self.llm is None:
                raise
            return self.llm

    def _execute_once(
        self,
        configuration: Configuration,
        document_text: str,
    ) -> tuple[ConOpsAnalysis, EvaluationReport, str, float, float]:
        started_at = perf_counter()
        active_llm = self._llm_for(configuration)
        model_name = getattr(active_llm, "model", active_llm.__class__.__name__)
        analysis = ConOpsExtractor(active_llm, rag=self.rag).extract(
            document_text,
            use_template=configuration.use_template,
            use_rules=configuration.use_rules,
            use_rag=configuration.use_rag,
            configuration_name=configuration.code,
            knowledge_format=self.knowledge_format,
        )
        rag_context = str(analysis.raw_notes.get("rag_context", ""))
        report = ConOpsEvaluator(
            active_llm if configuration.use_llm_evaluator else None,
            evaluator_strictness=self.evaluator_strictness,
        ).evaluate(
            analysis,
            document_text,
            configuration.code,
            self.reference_text,
            rag_context=rag_context,
            knowledge_format=self.knowledge_format,
        )
        sysml_score, _ = validate_sysml_text(generate_sysml_v2(analysis), analysis)
        return (
            analysis,
            report,
            model_name,
            round(perf_counter() - started_at, 3),
            sysml_score,
        )

    def run(
        self,
        document_text: str,
        selected_codes: list[str] | None = None,
        document_name: str = "document.pdf",
        n_repeats: int = 1,
        experiment_type: str = "strategy_comparison",
        fixed_variable: str = "",
        variable_tested: str = "strategy",
    ) -> tuple[
        list[ExperimentResult],
        dict[str, ConOpsAnalysis],
        dict[str, EvaluationReport],
    ]:
        if n_repeats < 1 or n_repeats > 10:
            raise ValueError("n_repeats hors intervalle autorise: 1 a 10")
        configurations = [
            item
            for item in CONFIGURATIONS
            if selected_codes is None or item.code in selected_codes
        ]
        self.run_id = uuid4().hex
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.document_hash = _hash_payload(document_text)
        self.errors = {}
        self.rag_rows = []
        results: list[ExperimentResult] = []
        analyses: dict[str, ConOpsAnalysis] = {}
        reports: dict[str, EvaluationReport] = {}

        for configuration in configurations:
            if self.rag:
                self.rag.clear_retrieval_log()
            config_payload = {
                **configuration.__dict__,
                "provider": self.provider,
                "model": self.models_by_configuration.get(configuration.code),
                "n_repeats": n_repeats,
                "temperature": 0.0,
                "evaluator_strictness": self.evaluator_strictness,
            }
            config_hash = _hash_payload(config_payload)
            prompt_hash = _hash_payload(
                {"document_hash": self.document_hash, **config_payload}
            )
            if configuration.uses_finetuned_model:
                readiness = fine_tuning_readiness()
            else:
                readiness = None
            if configuration.uses_finetuned_model and not readiness.is_ready:
                message = (
                    "Configuration preparee / non executee: protocole "
                    f"fine-tuning non eligible. {readiness.message()}"
                )
                self.errors[configuration.code] = message
                results.append(
                    ExperimentResult(
                        run_id=self.run_id,
                        timestamp=self.timestamp,
                        document_name=document_name,
                        document_hash=self.document_hash,
                        prompt_hash=prompt_hash,
                        config_hash=config_hash,
                        configuration=configuration.code,
                        description=configuration.description,
                        experiment_type=experiment_type,
                        fixed_variable=fixed_variable,
                        variable_tested=variable_tested,
                        strategy_name=configuration.code,
                        knowledge_format=self.knowledge_format,
                        provider=self.provider or "unknown",
                        model_name=str(readiness.adapter_path),
                        rag_used=False,
                        rules_used=configuration.use_rules,
                        evaluator_used=configuration.use_llm_evaluator,
                        llm_evaluator_used=configuration.use_llm_evaluator,
                        fine_tuned_used=False,
                        status="not_executed",
                        error_message=message,
                        repeat_count=n_repeats,
                        execution_notes=(
                            "C6/C7 ne sont comparables scientifiquement que si "
                            "un adaptateur LoRA, un dataset expert suffisant et "
                            "une evaluation contre C1/C3/C4/C5 sont disponibles."
                        ),
                    )
                )
                continue
            successes = []
            failures = []
            for _repeat_id in range(n_repeats):
                try:
                    successes.append(
                        self._execute_once(configuration, document_text)
                    )
                except Exception as exc:
                    failures.append(exc)

            if not successes:
                if self.rag:
                    self.rag_rows.extend(
                        {"configuration": configuration.code, **row}
                        for row in self.rag.retrieval_log
                    )
                failure_messages = [str(error) for error in failures]
                error_message = " | ".join(dict.fromkeys(failure_messages))
                failed_raw_response = next(
                    (
                        str(getattr(error, "raw_response", ""))
                        for error in failures
                        if getattr(error, "raw_response", "")
                    ),
                    "",
                )
                self.errors[configuration.code] = error_message
                results.append(
                    ExperimentResult(
                        run_id=self.run_id,
                        timestamp=self.timestamp,
                        document_name=document_name,
                        document_hash=self.document_hash,
                        prompt_hash=prompt_hash,
                        config_hash=config_hash,
                        configuration=configuration.code,
                        description=configuration.description,
                        experiment_type=experiment_type,
                        fixed_variable=fixed_variable,
                        variable_tested=variable_tested,
                        strategy_name=configuration.code,
                        knowledge_format=self.knowledge_format,
                        provider=self.provider or "unknown",
                        model_name=self.models_by_configuration.get(
                            configuration.code,
                            "unknown",
                        ),
                        rag_used=configuration.use_rag,
                        rules_used=configuration.use_rules,
                        evaluator_used=configuration.use_llm_evaluator,
                        llm_evaluator_used=configuration.use_llm_evaluator,
                        fine_tuned_used=configuration.uses_finetuned_model,
                        status="failed",
                        error_message=error_message,
                        json_valid=False,
                        json_validity_rate=0.0,
                        schema_compliance_rate=0.0,
                        json_error_message=error_message,
                        failed_raw_response=failed_raw_response,
                        repeat_count=n_repeats,
                    )
                )
                continue

            scores = [item[1].final_score for item in successes]
            durations = [item[3] for item in successes]
            score_mean = mean(scores)
            score_std = pstdev(scores) if len(scores) > 1 else 0.0
            stability = max(0.0, 100.0 - score_std)
            representative = min(
                successes,
                key=lambda item: abs(item[1].final_score - score_mean),
            )
            analysis, report, model_name, _, sysml_score = representative
            rag_stats = self.rag.retrieval_stats() if self.rag else {}
            actual_rag_used = bool(
                configuration.use_rag
                and rag_stats.get("retrieved_chunks_count", 0)
            )
            if self.rag:
                self.rag_rows.extend(
                    {"configuration": configuration.code, **row}
                    for row in self.rag.retrieval_log
                )
            report = report.model_copy(
                update={
                    "final_score": round(score_mean, 2),
                    "verdict": get_verdict(
                        score_mean,
                        report.human_review_rate,
                        report.classification_correctness_score,
                    ),
                }
            )
            analyses[configuration.code] = analysis
            reports[configuration.code] = report
            conops_metrics = [
                metric for metric in report.metrics if "SysML v2" not in metric.name
            ]
            conops_score = sum(
                metric.score * metric.weight for metric in conops_metrics
            ) / max(0.01, sum(metric.weight for metric in conops_metrics))
            notes = ""
            if failures:
                notes = f"{len(failures)} repetition(s) en echec sur {n_repeats}."
            if configuration.uses_finetuned_model:
                notes += " Fine-tuning declare; verifier le chargement LoRA reel."
            results.append(
                ExperimentResult(
                    run_id=self.run_id,
                    timestamp=self.timestamp,
                    document_name=document_name,
                    document_hash=self.document_hash,
                    prompt_hash=prompt_hash,
                    config_hash=config_hash,
                    configuration=configuration.code,
                    description=configuration.description,
                    experiment_type=experiment_type,
                    fixed_variable=fixed_variable,
                    variable_tested=variable_tested,
                    strategy_name=configuration.code,
                    knowledge_format=self.knowledge_format,
                    provider=self.provider or representative[0].__class__.__name__,
                    model_name=model_name,
                    rag_used=actual_rag_used,
                    rules_used=configuration.use_rules,
                    evaluator_used=configuration.use_llm_evaluator,
                    llm_evaluator_used=configuration.use_llm_evaluator,
                    fine_tuned_used=configuration.uses_finetuned_model,
                    status="success",
                    execution_time_sec=round(mean(durations), 3),
                    repeat_count=n_repeats,
                    mean_score=round(score_mean, 2),
                    std_score=round(score_std, 3),
                    stability_score=round(stability, 2),
                    final_score=round(score_mean, 2),
                    conops_score=round(conops_score, 2),
                    opscon_score=report.opscon_rule_score,
                    sysml_score=round(sysml_score * 100, 2),
                    semantic_similarity=report.semantic_similarity,
                    hallucination_level=report.hallucination_level,
                    hallucination_score=round(1 - report.grounding_score, 3),
                    grounding_score=report.grounding_score,
                    classification_correctness_score=(
                        report.classification_correctness_score
                    ),
                    concept_coverage_score=report.concept_coverage_score,
                    ontology_coverage_score=report.ontology_coverage_score,
                    rag_usage_score=report.rag_usage_score,
                    rules_version=report.rules_version,
                    rules_integrity_valid=True,
                    classification_rule_based=report.rule_based_classification,
                    rule_based_classification=report.rule_based_classification,
                    conops_rule_score=report.conops_rule_score,
                    opscon_rule_score=report.opscon_rule_score,
                    hybrid_score=min(
                        report.conops_rule_score or 0.0,
                        report.opscon_rule_score or 0.0,
                    ),
                    neither_score=max(
                        0.0,
                        100.0
                        - max(
                            report.conops_rule_score or 0.0,
                            report.opscon_rule_score or 0.0,
                        ),
                    ),
                    rule_compliance_score=max(
                        report.conops_rule_score or 0.0,
                        report.opscon_rule_score or 0.0,
                    ),
                    conops_forbidden_penalty=report.conops_forbidden_penalty,
                    opscon_forbidden_penalty=report.opscon_forbidden_penalty,
                    forbidden_penalty_total=report.forbidden_penalty,
                    forbidden_penalty=report.forbidden_penalty,
                    mandatory_missing_count=(
                        report.missing_mandatory_rules_count
                    ),
                    missing_mandatory_rules_count=(
                        report.missing_mandatory_rules_count
                    ),
                    opscon_missing_mandatory_penalty=(
                        report.opscon_missing_mandatory_penalty
                    ),
                    rule_confidence_score=report.classification_correctness_score,
                    rag_traceability_score=rag_stats.get(
                        "rag_traceability_score",
                        0.0,
                    ),
                    rule_evidence_coverage_score=rag_stats.get(
                        "rule_evidence_coverage_score",
                        0.0,
                    ),
                    json_validity_rate=1.0,
                    schema_compliance_rate=1.0,
                    json_valid=bool(analysis.raw_notes.get("json_valid", True)),
                    json_repair_attempts=int(
                        analysis.raw_notes.get("json_repair_attempts", 0)
                    ),
                    json_error_message=str(
                        analysis.raw_notes.get("json_error_message", "")
                    ),
                    evidence_coverage_score=report.evidence_coverage_score,
                    retrieved_chunks_count=rag_stats.get(
                        "retrieved_chunks_count", 0
                    ),
                    mean_retrieval_score=rag_stats.get(
                        "mean_retrieval_score", 0.0
                    ),
                    top_chunk_source=rag_stats.get("top_chunk_source", ""),
                    top_chunk_page=rag_stats.get("top_chunk_page"),
                    rag_context_tokens=rag_stats.get("rag_context_tokens", 0),
                    retrieval_hit_rate=rag_stats.get(
                        "retrieval_hit_rate",
                        0.0,
                    ),
                    chunks_used_in_answer=rag_stats.get(
                        "chunks_used_in_answer",
                        0,
                    ),
                    evidence_from_rag_count=rag_stats.get(
                        "evidence_from_rag_count",
                        0,
                    ),
                    repeat_index=0,
                    temperature=0.0,
                    human_review_count=report.human_review_count,
                    human_review_rate=report.human_review_rate,
                    sysml_validation_type=report.sysml_validation_type,
                    requirements_count=len(analysis.requirements),
                    stakeholders_count=len(analysis.stakeholders),
                    risks_count=len(analysis.risks),
                    interfaces_count=len(analysis.interfaces),
                    future_actions_count=len(analysis.future_actions),
                    evidence_count=self._evidence_count(analysis),
                    sysml_elements_count=self._sysml_element_count(analysis),
                    execution_notes=notes.strip(),
                )
            )

        successful = [result for result in results if result.status == "success"]
        if successful:
            max(successful, key=lambda item: item.final_score or 0).best_configuration = True
        return results, analyses, reports

    def run_strategy_comparison(
        self,
        document_text: str,
        *,
        fixed_model: str = "mistral:latest",
        document_name: str = "document.pdf",
        n_repeats: int = 1,
    ):
        codes = ["C1", "C3", "C4", "C5"]
        self.models_by_configuration = {code: fixed_model for code in codes}
        return self.run(
            document_text,
            selected_codes=codes,
            document_name=document_name,
            n_repeats=n_repeats,
            experiment_type="strategy_comparison",
            fixed_variable=f"model={fixed_model}",
            variable_tested="strategy",
        )

    def run_model_comparison(
        self,
        document_text: str,
        *,
        fixed_strategy: str = "C5",
        models: list[str] | None = None,
        document_name: str = "document.pdf",
        n_repeats: int = 1,
    ):
        models = models or [
            "llama3.2:latest",
            "mistral:latest",
            "qwen2.5:3b",
        ]
        combined_results = []
        combined_analyses = {}
        combined_reports = {}
        combined_rag_rows = []
        shared_run_id = uuid4().hex
        for index, model in enumerate(models, start=1):
            self.models_by_configuration = {fixed_strategy: model}
            results, analyses, reports = self.run(
                document_text,
                selected_codes=[fixed_strategy],
                document_name=document_name,
                n_repeats=n_repeats,
                experiment_type="model_comparison",
                fixed_variable=f"strategy={fixed_strategy}",
                variable_tested="model",
            )
            label = f"M{index}"
            combined_rag_rows.extend(
                {**row, "configuration": label}
                for row in self.rag_rows
            )
            for result in results:
                result.run_id = shared_run_id
                result.configuration = label
                result.strategy_name = fixed_strategy
            if fixed_strategy in analyses:
                combined_analyses[label] = analyses[fixed_strategy]
                combined_reports[label] = reports[fixed_strategy].model_copy(
                    update={"configuration": label}
                )
            combined_results.extend(results)
        self.run_id = shared_run_id
        self.rag_rows = combined_rag_rows
        successful = [item for item in combined_results if item.status == "success"]
        if successful:
            for item in combined_results:
                item.best_configuration = False
            max(successful, key=lambda item: item.final_score or 0).best_configuration = True
        return combined_results, combined_analyses, combined_reports

    def run_knowledge_format_comparison(
        self,
        document_text: str,
        *,
        fixed_model: str = "mistral:latest",
        fixed_strategy: str = "C5",
        document_name: str = "document.pdf",
        n_repeats: int = 1,
    ):
        combined_results = []
        combined_analyses = {}
        combined_reports = {}
        combined_rag_rows = []
        shared_run_id = uuid4().hex
        for label in ("K1", "K2", "K3"):
            self.knowledge_format = label
            self.models_by_configuration = {fixed_strategy: fixed_model}
            results, analyses, reports = self.run(
                document_text,
                selected_codes=[fixed_strategy],
                document_name=document_name,
                n_repeats=n_repeats,
                experiment_type="knowledge_comparison",
                fixed_variable=(
                    f"model={fixed_model};strategy={fixed_strategy}"
                ),
                variable_tested="knowledge_format",
            )
            for result in results:
                result.run_id = shared_run_id
                result.configuration = label
                result.strategy_name = fixed_strategy
                result.knowledge_format = label
            combined_rag_rows.extend(
                {**row, "configuration": label}
                for row in self.rag_rows
            )
            if fixed_strategy in analyses:
                combined_analyses[label] = analyses[fixed_strategy]
                combined_reports[label] = reports[fixed_strategy].model_copy(
                    update={"configuration": label}
                )
            combined_results.extend(results)
        self.run_id = shared_run_id
        self.rag_rows = combined_rag_rows
        successful = [item for item in combined_results if item.status == "success"]
        if successful:
            for item in combined_results:
                item.best_configuration = False
            max(successful, key=lambda item: item.final_score or 0).best_configuration = True
        return combined_results, combined_analyses, combined_reports


def results_to_dataframe(results: list[ExperimentResult]) -> pd.DataFrame:
    return pd.DataFrame([result.model_dump() for result in results])
