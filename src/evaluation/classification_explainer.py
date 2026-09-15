from __future__ import annotations


def explain_classification(
    rule_results: dict,
    *,
    document_title: str = "",
) -> dict:
    classification = rule_results.get("classification_by_rules", "Neither")
    rows = rule_results.get("rule_assessment_detailed", [])
    present = [row for row in rows if row.get("status") == "present"]
    missing = [row for row in rows if row.get("status") == "missing"]
    forbidden = [
        row for row in rows if row.get("status") == "forbidden_present"
    ]
    present_ids = [row["rule_id"] for row in present]
    missing_opscon = [
        row["rule_id"] for row in missing if row.get("rule_family") == "OpsCon"
    ]
    bullets = [
        f"Classification: {classification}.",
        f"Score ConOps: {rule_results.get('conops_rule_score', 0.0)}/100.",
        f"Score OpsCon: {rule_results.get('opscon_rule_score', 0.0)}/100.",
    ]
    if "D1" in present_ids:
        bullets.append("D1 Problem Space orientation est present.")
    if "D2" not in present_ids:
        bullets.append("D2 Solution Space orientation est faible ou absent.")
    if present_ids:
        bullets.append(
            "Regles presentes: " + ", ".join(present_ids[:12]) + "."
        )
    if missing_opscon:
        bullets.append(
            "Regles OpsCon absentes: " + ", ".join(missing_opscon[:8]) + "."
        )
    if forbidden:
        bullets.append(
            "Violations interdites detectees: "
            + ", ".join(row["rule_id"] for row in forbidden[:8])
            + "."
        )
    lower_title = document_title.lower()
    if "opscon" in lower_title and classification == "ConOps":
        bullets.append(
            "Contradiction titre/contenu: le titre suggere OpsCon mais la "
            "structure evaluee correspond davantage a un ConOps."
        )
    if classification == "Hybrid":
        bullets.append(
            "Document hybride: le document repond a la fois a pourquoi changer "
            "et a comment le futur systeme operationnel fonctionnera."
        )
    bullets.append("Validation expert recommandee.")
    return {
        "classification": classification,
        "bullets": bullets,
        "summary": " ".join(bullets),
    }
