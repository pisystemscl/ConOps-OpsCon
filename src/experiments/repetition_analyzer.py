from __future__ import annotations

from collections import Counter
import math
from statistics import mean


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value or default)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _population_stddev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    average = mean(values)
    variance = sum((value - average) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def analyze_repetitions(results: list[dict]) -> dict:
    if not results:
        return {
            "mean_score": 0.0,
            "std_score": 0.0,
            "stability_score": 0.0,
            "json_validity_rate": 0.0,
            "rule_score_consistency": 0.0,
            "classification_consistency": 0.0,
            "extraction_consistency_score": 0.0,
        }
    scores = [_safe_float(item.get("final_score")) for item in results]
    conops_scores = [
        _safe_float(item.get("conops_rule_score")) for item in results
    ]
    classifications = [
        str(
            item.get("classification_rule_based")
            or item.get("rule_based_classification")
            or ""
        )
        for item in results
    ]
    json_valid = [bool(item.get("json_valid", True)) for item in results]
    std_score = _population_stddev(scores)
    mean_score = mean(scores)
    rule_std = _population_stddev(conops_scores)
    most_common_classification = Counter(classifications).most_common(1)[0][1]
    return {
        "mean_score": round(mean_score, 3),
        "std_score": round(std_score, 3),
        "min_score": round(min(scores), 3),
        "max_score": round(max(scores), 3),
        "stability_score": round(max(0.0, 100.0 - std_score), 3),
        "coefficient_of_variation": round(std_score / mean_score, 4)
        if mean_score
        else 0.0,
        "json_validity_rate": round(sum(json_valid) / len(json_valid), 3),
        "rule_score_consistency": round(max(0.0, 100.0 - rule_std), 3),
        "classification_consistency": round(
            most_common_classification / len(classifications),
            3,
        ),
        "extraction_consistency_score": round(max(0.0, 100.0 - std_score), 3),
    }


def repetition_warning(repeat_count: int) -> str:
    if repeat_count <= 1:
        return (
            "Une seule repetition ne suffit pas pour une conclusion "
            "scientifique robuste."
        )
    if repeat_count < 5:
        return "Repeat_count minimum recommande = 5 pour une analyse robuste."
    return ""
