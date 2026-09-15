import pandas as pd
import pyarrow as pa

from app import (
    automatic_scientific_conclusion,
    build_configuration_delta_table,
    build_configuration_kpi_table,
    dataframe_for_display,
)


def test_dataframe_for_display_serializes_nested_cells():
    frame = dataframe_for_display(
        [
            {
                "expected": ["ConOps"],
                "matched_terms": ["stakeholder needs", "future services"],
                "status": "present",
            },
            {
                "expected": "mandatory",
                "matched_terms": [],
                "status": "missing",
            },
        ]
    )
    assert isinstance(frame, pd.DataFrame)
    assert frame.loc[0, "expected"] == '["ConOps"]'
    assert frame.loc[1, "expected"] == "mandatory"
    assert frame.loc[1, "matched_terms"] == "[]"
    assert pa.Table.from_pandas(frame) is not None


def test_dataframe_for_display_serializes_mixed_object_columns():
    frame = dataframe_for_display(
        pd.DataFrame(
            [
                ("Document analysé", "sample.pdf"),
                ("Nombre de répétitions", 1),
                ("Backend", b"local"),
            ],
            columns=["Paramètre", "Valeur"],
        )
    )

    assert frame["Valeur"].tolist() == ["sample.pdf", "1", "local"]
    assert pa.Table.from_pandas(frame) is not None


def test_automatic_scientific_conclusion_handles_missing_values():
    frame = pd.DataFrame(
        [
            {
                "configuration": "C1",
                "final_score": 50,
                "grounding_score": None,
                "semantic_similarity": None,
                "human_review_rate": None,
            },
            {
                "configuration": "C3",
                "final_score": 55,
                "grounding_score": 0.0,
                "semantic_similarity": 0.0,
                "human_review_rate": None,
            },
        ]
    )

    conclusion = automatic_scientific_conclusion(frame)

    assert "La meilleure configuration finale est C3" in conclusion


def test_configuration_kpi_table_is_specific_and_readable():
    table = build_configuration_kpi_table(
        {
            "configuration": "C4",
            "final_score": 72,
            "mean_score": 72,
            "std_score": 2,
            "stability_score": 98,
            "json_validity_rate": 1,
            "grounding_score": 0.8,
            "repeat_count": 5,
        }
    )

    assert table.columns.tolist() == [
        "Famille",
        "KPI / métrique",
        "Valeur",
        "Unité / lecture",
    ]
    assert table.loc[table["KPI / métrique"] == "Taux de JSON valide", "Valeur"].iloc[0] == 100
    assert table.loc[table["KPI / métrique"] == "Coefficient de variation", "Valeur"].iloc[0] == 0.0278


def test_configuration_delta_table_quantifies_each_component():
    table = build_configuration_delta_table(
        pd.DataFrame(
            [
                {"configuration": "C1", "final_score": 50, "grounding_score": 0.4},
                {"configuration": "C3", "final_score": 60, "grounding_score": 0.5},
                {"configuration": "C4", "final_score": 65, "grounding_score": 0.8},
                {"configuration": "C5", "final_score": 63, "grounding_score": 0.85},
            ]
        )
    )

    assert table["Comparaison"].tolist() == ["C3 − C1", "C4 − C3", "C5 − C4"]
    assert table["Δ score final"].tolist() == [10.0, 5.0, -2.0]
