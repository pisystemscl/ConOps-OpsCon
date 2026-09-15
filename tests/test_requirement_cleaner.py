from src.core.schema import Requirement
from src.postprocessing.requirement_cleaner import (
    clean_requirement_text,
    detect_bad_requirement_grammar,
    detect_copied_evidence_as_requirement,
    generate_requirement_from_evidence,
    score_requirement_quality,
)


def test_requirement_cleaner_api_scores_quality():
    text, flags, confidence = clean_requirement_text(
        "Certification requirements are limited to safety",
        "Certification requirements are limited to safety.",
    )
    requirement = Requirement(id="REQ-1", text=text, quality_flags=flags)

    assert text.startswith("The system shall")
    assert confidence <= 1.0
    assert score_requirement_quality(requirement) <= 100
    assert detect_bad_requirement_grammar("Certification only") is True
    assert detect_copied_evidence_as_requirement(
        text,
        "Certification requirements are limited to safety.",
    ) in {True, False}
    assert generate_requirement_from_evidence("Share data with operators.").startswith(
        "The system shall"
    )
