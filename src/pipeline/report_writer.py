from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.schema import ConOpsAnalysis, EvaluationReport


def write_markdown_report(
    path: str | Path,
    analysis: ConOpsAnalysis,
    report: EvaluationReport,
    experiments_df: pd.DataFrame | None = None,
    run_id: str = "",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Rapport d'analyse ConOps - {analysis.document_title}", ""]
    if run_id:
        lines.append(f"**Run ID :** `{run_id}`")
    lines.extend(
        [
            "**Statut :** résultat validé automatiquement, non validé par un expert métier.",
            "**Validation SysML :** heuristique interne, non officielle.",
            "",
            f"**Score final :** {report.final_score}/100",
            f"**Verdict :** {report.verdict}",
            f"**Classification V3.0 :** {report.rule_based_classification}",
            f"**Score règles ConOps :** {report.conops_rule_score}/100",
            f"**Score règles OpsCon :** {report.opscon_rule_score}/100",
            (
                "**Pénalité des règles interdites ConOps :** "
                f"{report.conops_forbidden_penalty}"
            ),
            (
                "**Pénalité des règles interdites OpsCon :** "
                f"{report.opscon_forbidden_penalty}"
            ),
            f"**Hallucination :** {report.hallucination_level}",
            f"**Grounding documentaire :** {report.grounding_score}",
            (
                "**Exactitude de classification :** "
                f"{report.classification_correctness_score}"
            ),
            f"**Éléments à revoir :** {report.human_review_count}",
            f"**Taux de revue humaine :** {report.human_review_rate:.1%}",
            "",
            "## Conclusion de classification",
            (
                f"Le contenu est principalement classé "
                f"**{report.rule_based_classification}** par les règles V3.0."
            ),
            "",
            "## Résumé",
            analysis.summary or "Résumé non disponible.",
            "",
            "## Métriques",
        ]
    )
    for metric in report.metrics:
        lines.append(
            f"- **{metric.name}** : {metric.score:.1f}/100 "
            f"(poids={metric.weight:.2f}) - {metric.comment}"
        )
    lines.extend(["", "## Évaluation déterministe règle par règle"])
    for item in report.rule_assessment_detailed:
        contribution = item.get(
            "score_contribution",
            item.get(
                "conops_score_contribution",
                item.get("opscon_score_contribution", 0),
            ),
        )
        lines.append(
            f"- **{item['rule_id']} - {item['rule_name']}** : "
            f"{item['status']} (poids={item['weight']}, "
            f"contribution={contribution})"
        )
    lines.extend(["", "## Parties prenantes"])
    for stakeholder in analysis.stakeholders:
        lines.append(
            f"- **{stakeholder.name}** - rôle : {stakeholder.role}; "
            f"intérêt : {stakeholder.interest}"
        )
    for heading, values in (
        ("Services attendus", analysis.expected_services),
        ("Capabilities", analysis.capabilities),
        ("Enablers", analysis.enablers),
        ("Drivers", analysis.drivers),
        ("Mitigations", analysis.mitigations),
        ("Gaps", analysis.gaps),
    ):
        lines.extend(["", f"## {heading}"])
        lines.extend(f"- {value}" for value in values)
    lines.extend(["", "## Exigences haut niveau"])
    for requirement in analysis.requirements:
        lines.append(
            f"- **{requirement.id}** [{requirement.priority}] {requirement.text}"
        )
    lines.extend(["", "## Risques"])
    for risk in analysis.risks:
        lines.append(
            f"- {risk.description} | impact={risk.impact}, "
            f"probabilité={risk.probability}, mitigation={risk.mitigation}"
        )
    lines.extend(["", "## Limites et avertissements"])
    quality_flags = analysis.raw_notes.get("quality_flags", [])
    if quality_flags:
        for item in quality_flags:
            lines.append(
                f"- {item.get('item_id', 'item')} : "
                f"{', '.join(item.get('flags', []))}"
            )
    else:
        lines.append(
            "- Aucun drapeau automatique détecté; une revue humaine reste requise."
        )
    lines.extend(["", "## Recommandations"])
    lines.extend(f"- {recommendation}" for recommendation in report.recommendations)
    if experiments_df is not None and not experiments_df.empty:
        lines.extend(
            [
                "",
                "## Comparaison des configurations experimentales",
                experiments_df.to_markdown(index=False),
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_global_run_report(
    path: str | Path,
    experiments_df: pd.DataFrame,
    run_id: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Rapport global du run {run_id}",
        "",
        "Résultats validés automatiquement, non validés par un expert métier.",
        "",
        "## Formule du score",
        "",
        "Score final = 70% évaluation déterministe des règles V3.0 + "
        "30% contrôles techniques de qualité, grounding et SysML.",
        "",
        "## Comparaison",
        "",
        experiments_df.to_markdown(index=False),
        "",
        "## Limites",
        "",
        "- Les scores sont heuristiques et restent soumis a validation expert metier.",
        "- La validation SysML n'est pas une validation officielle.",
        "- Plusieurs documents et répétitions sont nécessaires pour généraliser.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
