from src.evaluation.section_classifier import classify_sections, split_document_sections


def test_section_classification_exports_rule_scores():
    text = (
        "[Page 1]\nExecutive Summary\nThe current situation includes stakeholder needs "
        "and motivation for change.\n"
        "[Page 2]\nOperational Realization\nThe future operational SoS includes "
        "constituent systems and operational flows."
    )

    sections = split_document_sections(text)
    frame = classify_sections(text)

    assert sections
    assert not frame.empty
    assert {"conops_score", "opscon_score", "classification"} <= set(frame.columns)
    assert "rules_present" in frame.columns
