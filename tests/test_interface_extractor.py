from src.extraction.interface_extractor import extract_structured_interfaces


def test_interface_extractor_structures_actor_flow_and_review_flags():
    result = extract_structured_interfaces(
        "[Page 5]\nANSPs exchange trajectory data with Network Manager.",
        classification="ConOps",
    )

    assert result.interfaces
    interface = result.interfaces[0]
    assert interface.source_actor == "ANSPs"
    assert interface.target_actor == "Network Manager"
    assert interface.flow_type == "Information"
    assert interface.confidence == 1.0
    assert interface.human_review_needed is False
    assert interface.evidence_quote
    assert interface.source_page == 5
    assert "POSSIBLE_OPSCON_DRIFT" in interface.quality_flags
    assert result.interface_completeness_score == 1.0
