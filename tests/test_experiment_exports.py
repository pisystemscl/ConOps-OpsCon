from pathlib import Path

import pandas as pd

from src.core.schema import EvaluationReport, ExperimentResult
from src.llm.providers import MockLLM
from src.pipeline.experiment_runner import ExperimentRunner
from src.pipeline.exporter import export_detailed_csv


def test_evaluation_report_accepts_structured_verdict():
    report = EvaluationReport.model_validate(
        {
            "configuration": "C5",
            "metrics": [],
            "final_score": 75,
            "verdict": {
                "completude": "Bonne",
                "hallucination": "Faible",
            },
            "recommendations": [],
            "hallucination_level": "faible",
            "semantic_similarity": 0.7,
            "grounding_score": 0.8,
            "sysml_validity_score": 0.9,
        }
    )

    assert report.verdict == "completude: Bonne; hallucination: Faible"


def test_runner_keeps_failed_configuration(monkeypatch):
    def failing_factory(code, provider=None, model=None):
        raise RuntimeError(f"{code} unavailable")

    monkeypatch.setattr(
        "src.pipeline.experiment_runner.get_llm_for_configuration",
        failing_factory,
    )
    runner = ExperimentRunner(provider="ollama")

    results, analyses, reports = runner.run(
        "A sufficiently long ConOps source document.",
        selected_codes=["C5"],
        document_name="sample.pdf",
    )

    assert len(results) == 1
    assert results[0].configuration == "C5"
    assert results[0].status == "failed"
    assert results[0].final_score is None
    assert "unavailable" in results[0].error_message
    assert analyses == {}
    assert reports == {}


def test_detailed_exports_are_created(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.pipeline.experiment_runner.get_llm_for_configuration",
        lambda code, provider=None, model=None: MockLLM(),
    )
    runner = ExperimentRunner(provider="mock")
    results, analyses, reports = runner.run(
        (
            "The future ATM network shall share flight information with ANSPs. "
            "Airspace users require resilient trajectory services. "
            "Cyber security risk requires mitigation and validation."
        ),
        selected_codes=["C1"],
        document_name="sample.pdf",
    )

    paths = export_detailed_csv(
        results,
        analyses,
        reports,
        tmp_path,
        file_prefix="test_run",
    )

    assert set(paths) == {
        "summary",
        "extractions",
        "criteria",
        "rag",
        "rules",
        "graph_edges",
    }
    assert all(path.exists() for path in paths.values())
    summary = pd.read_csv(paths["summary"])
    criteria = pd.read_csv(paths["criteria"])
    assert summary.loc[0, "run_id"] == runner.run_id
    assert summary.loc[0, "document_name"] == "sample.pdf"
    assert "human_review_count" in summary.columns
    assert "human_review_rate" in summary.columns
    assert "retrieved_chunks_count" in summary.columns
    assert not criteria.empty


def test_rag_export_contains_prompt_usage(tmp_path: Path):
    result = ExperimentResult(
        run_id="run-1",
        timestamp="2026-06-15T00:00:00Z",
        document_name="target.pdf",
        configuration="C4",
        description="LLM + RAG",
        provider="mock",
        model_name="mock",
        rag_used=True,
        rules_used=False,
        evaluator_used=False,
        fine_tuned_used=False,
        retrieved_chunks_count=1,
    )
    paths = export_detailed_csv(
        [result],
        {},
        {},
        tmp_path,
        rag_rows=[
            {
                "configuration": "C4",
                "query": "OpsCon interfaces",
                "chunk_id": "chunk-1",
                "source_document": "reference.pdf",
                "page": 4,
                "section": "Interfaces",
                "retrieval_score": 0.9,
                "distance": 0.1,
                "used_in_prompt": True,
                "chunk_text": "Operational interfaces support flows.",
                "source_path": "/reference.pdf",
            }
        ],
    )

    rag_frame = pd.read_csv(paths["rag"])
    summary = pd.read_csv(paths["summary"])
    assert not rag_frame.empty
    assert bool(rag_frame.loc[0, "used_in_prompt"])
    assert summary.loc[0, "retrieved_chunks_count"] == 1


def test_failed_json_response_is_exported(tmp_path: Path):
    result = ExperimentResult(
        run_id="run-1",
        timestamp="2026-06-15T00:00:00Z",
        document_name="target.pdf",
        configuration="C5",
        description="LLM + RAG + evaluateur",
        provider="mock",
        model_name="mock",
        rag_used=True,
        rules_used=True,
        evaluator_used=True,
        fine_tuned_used=False,
        status="failed",
        error_message="JSON LLM invalide",
        failed_raw_response='{"requirements":[{"priority":"high"}]}',
    )

    export_detailed_csv([result], {}, {}, tmp_path)

    failed_path = tmp_path / "failed_json_responses" / "C5_failed_response.txt"
    assert failed_path.exists()
    assert result.raw_response_path == str(failed_path)
    assert "requirements" in failed_path.read_text(encoding="utf-8")


def test_runner_computes_repeat_statistics(monkeypatch):
    monkeypatch.setattr(
        "src.pipeline.experiment_runner.get_llm_for_configuration",
        lambda code, provider=None, model=None: MockLLM(),
    )
    runner = ExperimentRunner(provider="mock")
    results, _, _ = runner.run(
        "The network shall exchange flight data. Cyberattack is a risk. " * 5,
        selected_codes=["C1"],
        n_repeats=3,
    )

    assert results[0].repeat_count == 3
    assert results[0].mean_score is not None
    assert results[0].std_score == 0
    assert results[0].stability_score == 100
