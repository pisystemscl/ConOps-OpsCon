from src.experiments.repetition_analyzer import analyze_repetitions, repetition_warning


def test_repetition_analyzer_computes_stability():
    metrics = analyze_repetitions(
        [
            {"final_score": 70, "conops_rule_score": 75, "json_valid": True, "classification_rule_based": "ConOps"},
            {"final_score": 72, "conops_rule_score": 76, "json_valid": True, "classification_rule_based": "ConOps"},
        ]
    )

    assert metrics["mean_score"] == 71
    assert metrics["classification_consistency"] == 1.0
    assert repetition_warning(1)


def test_repetition_analyzer_handles_non_finite_scores():
    metrics = analyze_repetitions(
        [
            {"final_score": 70.0, "conops_rule_score": 75.0},
            {"final_score": float("nan"), "conops_rule_score": float("inf")},
        ]
    )

    assert metrics["mean_score"] == 35.0
    assert metrics["std_score"] == 35.0
    assert metrics["rule_score_consistency"] == 62.5
