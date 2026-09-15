from src.core.schema import ConOpsAnalysis, Interface, Requirement
from src.llm.providers import MockLLM
from src.pipeline.experiment_runner import CONFIGURATIONS, ExperimentRunner
from src.pipeline.post_processor import enrich_analysis
from src.pipeline.sysml_generator import generate_sysml_v2, validate_sysml_text
from src.utils.io import safe_extract_json


def test_safe_extract_json_repairs_missing_comma_and_code_fence():
    answer = """```json
    {
      "stakeholders": [{"name": "ANSP"}]
      "requirements": [{"id": "REQ-001", "text": "Share flight data"}],
    }
    ```"""

    data = safe_extract_json(answer)

    assert data["stakeholders"][0]["name"] == "ANSP"
    assert data["requirements"][0]["id"] == "REQ-001"


def test_enrich_analysis_normalizes_requirement_and_adds_evidence():
    source = "Full sharing of relevant flight information with all network actors."
    analysis = ConOpsAnalysis(
        requirements=[
            Requirement(
                id="REQ-001",
                text="Full sharing of relevant flight information with all network actors",
            )
        ]
    )

    enriched = enrich_analysis(analysis, source)

    assert enriched.requirements[0].text.startswith("The system shall support ")
    assert enriched.requirements[0].evidence


def test_sysml_score_is_not_perfect_without_interfaces():
    analysis = ConOpsAnalysis(
        document_title="Test",
        document_classification="OpsCon",
        requirements=[Requirement(id="REQ-001", text="The system shall operate.")],
    )

    score, errors = validate_sysml_text(generate_sysml_v2(analysis))

    assert score < 1.0
    assert any("connexion" in error.lower() for error in errors)


def test_sysml_score_remains_heuristic_with_interface():
    analysis = ConOpsAnalysis(
        document_title="Test",
        document_classification="OpsCon",
        requirements=[Requirement(id="REQ-001", text="The system shall operate.")],
        interfaces=[
            Interface(
                source="ANSP",
                target="Network Manager",
                exchanged_information="Flight data",
            )
        ],
    )

    score, errors = validate_sysml_text(generate_sysml_v2(analysis))

    assert 0.8 <= score < 1.0
    assert errors == []


def test_runner_passes_selected_provider_and_model(monkeypatch):
    calls = []

    def fake_factory(code, provider=None, model=None):
        calls.append((code, provider, model))
        return MockLLM()

    monkeypatch.setattr(
        "src.pipeline.experiment_runner.get_llm_for_configuration",
        fake_factory,
    )
    runner = ExperimentRunner(
        provider="ollama",
        models_by_configuration={"C1": "qwen2.5:3b"},
    )

    runner._llm_for(
        next(configuration for configuration in CONFIGURATIONS if configuration.code == "C1")
    )

    assert calls == [("C1", "ollama", "qwen2.5:3b")]
