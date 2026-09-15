from __future__ import annotations

from src.domain.ontology_builder import build_ontology
from src.domain.rules_loader import all_rules

ALIASES = {
    "K1": "simple_list",
    "K2": "definitions",
    "K3": "ontology",
    "list": "simple_list",
}


def get_knowledge_context(format_name: str) -> str:
    format_name = ALIASES.get(format_name, format_name)
    rules = all_rules()
    if format_name == "simple_list":
        return "\n".join(f"- {rule['id']}: {rule['name']}" for rule in rules)
    if format_name == "definitions":
        return "Definitions\n" + "\n".join(
            f"- {rule['id']} {rule['name']}: {rule['description']}"
            for rule in rules
        )
    if format_name == "ontology":
        ontology = build_ontology()
        blocks = []
        for family, content in ontology.items():
            blocks.append(f"{family.upper()} CONCEPTS")
            blocks.extend(f"- {concept}" for concept in content["concepts"])
            blocks.append(f"{family.upper()} RELATIONS")
            blocks.extend(
                f"- {source} --{relation}--> {target}"
                for source, relation, target in content["relations"]
            )
        return "\n".join(blocks)
    raise ValueError(
        "Unknown knowledge format. Use K1/K2/K3 or "
        "simple_list/definitions/ontology."
    )
