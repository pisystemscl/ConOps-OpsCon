from __future__ import annotations

from src.domain.rules_loader import all_rules, build_rules_prompt_context, load_rules


def rules_as_text() -> str:
    return build_rules_prompt_context()


OFFICIAL_CONOPS_RULES = [
    (rule["id"], rule["name"])
    for rule in load_rules()["conops_rules"]
]

RULE_DEFINITIONS = {
    rule["name"]: rule["description"]
    for rule in all_rules()
}
