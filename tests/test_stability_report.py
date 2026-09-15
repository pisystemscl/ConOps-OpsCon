import pandas as pd

from src.evaluation.configuration_explainer import explain_best_configuration
from src.experiments.repetition_analyzer import analyze_repetitions, repetition_warning


def test_stability_report_and_best_configuration_explanation():
    rows = [
        {
            "configuration": "C1",
            "status": "success",
            "final_score": 70,
            "grounding_score": 0.4,
            "semantic_similarity": 0.5,
            "rag_used": False,
            "llm_evaluator_used": False,
        },
        {
            "configuration": "C5",
            "status": "success",
            "final_score": 72,
            "grounding_score": 0.8,
            "semantic_similarity": 0.6,
            "rag_used": True,
            "llm_evaluator_used": True,
        },
    ]
    frame = pd.DataFrame(rows)
    metrics = analyze_repetitions(rows)
    explanation = explain_best_configuration(frame, "C5")

    assert metrics["json_validity_rate"] == 1.0
    assert explanation["gain_vs_C1"] == 2
    assert any("Gain faible" in line for line in explanation["bullets"])
    assert "Une seule repetition" in repetition_warning(1)
    assert "minimum recommande" in repetition_warning(3)
