from pathlib import Path

import pandas as pd

from src.core.schema import (
    ConOpsAnalysis,
    Evidence,
    ExperimentResult,
    Interface,
    Requirement,
    Stakeholder,
)
from src.evaluation.baseline_gain import (
    gain_vs_baseline,
    interpret_baseline_gain,
)
from src.evaluation.rule_based_evaluator import (
    compute_rule_confidence,
    evaluate_document_rules,
)
from src.pipeline.batch_document_runner import BatchDocumentRunner
from src.postprocessing.requirement_cleaner import clean_requirement
from src.rules.rules_loader import (
    get_active_rules,
    get_rule_by_id,
    get_rules_by_family,
)


def test_official_rule_lookup_api_is_complete():
    assert get_rule_by_id("D1")["family"] == "D"
    assert len(get_rules_by_family("ConOps")) == 14
    assert len(get_rules_by_family("OpsCon")) == 15
    assert len(get_active_rules()) == 31


def test_requirement_cleaner_separates_reformulation_from_evidence():
    evidence = Evidence(
        text="The difference is that safe pad operations are required.",
        page=38,
    )
    requirement = clean_requirement(
        Requirement(
            id="REQ-1",
            text=(
                "The difference is that safe pad operations must be "
                "supported for every future passenger mission"
            ),
            evidence=[evidence],
        )
    )

    assert requirement.text.startswith("The system shall ")
    assert len(requirement.text.split()) <= 30
    assert requirement.evidence[0].text == evidence.text
    assert "MISSING_EVIDENCE" not in requirement.quality_flags


def test_interface_exposes_scientific_fields_and_missing_actor_flags():
    interface = Interface(
        interface_id="INT-1",
        source_actor="Operator",
        exchanged_item="Certification data",
        flow_type="Information",
    )

    assert interface.source == "Operator"
    assert interface.exchanged_information == "Certification data"
    assert interface.target_actor == ""


def test_rule_evaluation_exposes_compliance_matrix_and_confidence():
    analysis = ConOpsAnalysis(
        document_classification="OpsCon",
        stakeholders=[Stakeholder(name="Airspace Users")],
        requirements=[
            Requirement(
                id="REQ-1",
                text="The system shall support safe UAM port operations.",
            )
        ],
    )
    analysis.raw_notes["source_text"] = (
        "[Page 4]\nThe current situation describes stakeholder needs and "
        "desired future outcomes.\n[Page 8]\nOperational flows allocate "
        "services to constituent systems."
    )
    report = evaluate_document_rules(analysis)

    assert report["rules_version"] == "3.0"
    assert report["rule_assessment_detailed"]
    assert 0 <= compute_rule_confidence(report) <= 1


def test_gain_vs_c1_and_low_gain_interpretation():
    frame = pd.DataFrame(
        [
            {
                "configuration": "C1",
                "status": "success",
                "final_score": 70.0,
                "conops_score": 70,
                "opscon_score": 50,
                "sysml_score": 60,
                "semantic_similarity": 0.7,
                "grounding_score": 0.5,
                "rag_used": False,
                "llm_evaluator_used": False,
                "human_review_rate": 0.3,
            },
            {
                "configuration": "C5",
                "status": "success",
                "final_score": 71.0,
                "conops_score": 71,
                "opscon_score": 55,
                "sysml_score": 65,
                "semantic_similarity": 0.72,
                "grounding_score": 0.7,
                "rag_used": True,
                "llm_evaluator_used": True,
                "human_review_rate": 0.2,
            },
        ]
    )

    gain = gain_vs_baseline(frame)
    assert gain.loc[gain["configuration"] == "C5", "gain_vs_C1"].iloc[0] == 1
    assert any(
        "Gain faible" in message
        for message in interpret_baseline_gain(frame)
    )


def _result(document_name: str, configuration: str, score: float):
    return ExperimentResult(
        run_id="run",
        timestamp="2026-06-15T00:00:00Z",
        document_name=document_name,
        configuration=configuration,
        description=configuration,
        provider="mock",
        model_name="mock",
        rag_used=configuration in {"C4", "C5"},
        rules_used=configuration in {"C3", "C5"},
        evaluator_used=configuration == "C5",
        llm_evaluator_used=configuration == "C5",
        fine_tuned_used=False,
        final_score=score,
        grounding_score=0.8,
        human_review_rate=0.2,
    )


def test_batch_runner_aggregates_documents():
    class FakeRunner:
        def run_strategy_comparison(
            self,
            document_text,
            *,
            fixed_model,
            document_name,
            n_repeats,
        ):
            offset = 1 if "second" in document_text else 0
            return (
                [
                    _result(document_name, "C1", 70 + offset),
                    _result(document_name, "C5", 80 + offset),
                ],
                {},
                {},
            )

    result = BatchDocumentRunner(lambda: FakeRunner()).run(
        [("one.pdf", "first"), ("two.pdf", "second")],
        fixed_model="mock",
    )

    assert result.best_configuration == "C5"
    assert len(result.by_document) == 4
    assert set(result.by_configuration["configuration"]) == {"C1", "C5"}
