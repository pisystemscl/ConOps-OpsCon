from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_scientific_summary(
    output_path: str | Path,
    *,
    document_name: str,
    contradiction_detected: bool,
    final_classification: str,
    results: pd.DataFrame,
    best_explanation: dict,
    section_frame: pd.DataFrame,
    stability_report: dict,
    rag_rows_count: int,
) -> Path:
    path = Path(output_path)
    lines = [
        "# Scientific Summary",
        "",
        f"Document analyse: {document_name}",
        f"Classification finale: {final_classification}",
        "",
        "## Contradiction titre/contenu",
        (
            "Contradiction detectee: le titre suggere OpsCon mais le contenu est majoritairement ConOps."
            if contradiction_detected
            else "Aucune contradiction titre/contenu majeure detectee."
        ),
        "",
        "## Resultats par configuration",
        "```text",
        results.to_string(index=False) if not results.empty else "Aucun resultat.",
        "```",
        "",
        "## Pourquoi cette configuration gagne ?",
        *[f"- {line}" for line in best_explanation.get("bullets", [])],
        "",
        "## Analyse RAG",
        f"Nombre de chunks traces: {rag_rows_count}",
        "",
        "## Analyse par sections",
        "```text",
        section_frame.head(20).to_string(index=False)
        if not section_frame.empty
        else "Aucune section detectee.",
        "```",
        "",
        "## Analyse de stabilite",
        "```json",
        str(stability_report),
        "```",
        "",
        "## Limites",
        "- Evaluation automatique a confirmer par validation expert.",
        "- Resultats a confirmer sur plusieurs documents et repetitions.",
        "- SysML v2 genere automatiquement avec validation heuristique non officielle.",
        "",
        "## Conclusion scientifique",
        "La plateforme combine LLM, RAG Chroma, regles officielles V3.0, stabilite experimentale et traceabilite documentaire.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
