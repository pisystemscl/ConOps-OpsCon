from src.core.schema import ConOpsAnalysis


def test_analysis_normalizes_object_items_in_text_lists():
    analysis = ConOpsAnalysis.model_validate(
        {
            "as_is": [
                {"description": "Système actuel"},
                {"text": "Procédure manuelle"},
            ],
            "gaps": {"description": "Traçabilité insuffisante"},
            "things_to_change": [
                {"description": "Automatiser l'évaluation", "page": None},
            ],
            "things_to_avoid": [
                "Inventer des exigences",
                {"content": "Confondre ConOps et OpsCon"},
            ],
        }
    )

    assert analysis.as_is == ["Système actuel", "Procédure manuelle"]
    assert analysis.gaps == ["Traçabilité insuffisante"]
    assert analysis.things_to_change == ["Automatiser l'évaluation"]
    assert analysis.things_to_avoid == [
        "Inventer des exigences",
        "Confondre ConOps et OpsCon",
    ]


def test_analysis_normalizes_nested_ollama_variants():
    analysis = ConOpsAnalysis.model_validate(
        {
            "stakeholders": [
                {
                    "name": "Airspace User",
                    "evidence": ["Supporting Airspace User business objectives"],
                }
            ],
            "requirements": [
                {
                    "id": "REQ-001",
                    "text": "Support the Single European Sky",
                    "evidence": "Supporting the Single European Sky",
                }
            ],
            "risks": [
                {
                    "description": "Scalability risk",
                    "mitigation": [
                        "Harmonisation of procedures",
                        "Standardisation of systems",
                    ],
                    "evidence": ["Scalability solutions need to be put in place"],
                }
            ],
            "future_actions": [
                "Review local operations",
                {"action": "Validate target solutions", "owner": "AMAN"},
            ],
        }
    )

    assert analysis.stakeholders[0].evidence[0].text == (
        "Supporting Airspace User business objectives"
    )
    assert analysis.requirements[0].evidence[0].source == "document_utilisateur"
    assert analysis.risks[0].mitigation == (
        "Harmonisation of procedures; Standardisation of systems"
    )
    assert analysis.risks[0].evidence[0].text == (
        "Scalability solutions need to be put in place"
    )
    assert analysis.future_actions[0].action == "Review local operations"


def test_analysis_coerces_numeric_stakeholder_influence_level():
    analysis = ConOpsAnalysis.model_validate(
        {
            "stakeholders": [
                {
                    "name": "Network Manager",
                    "role": "Coordinator",
                    "influence_level": 0.8,
                }
            ]
        }
    )

    assert analysis.stakeholders[0].influence_level == "0.8"
