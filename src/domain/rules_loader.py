from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

APPLICATION_DIR = Path(__file__).resolve().parents[2]
RULES_PATH = APPLICATION_DIR / "data" / "rules" / "conops_opscon_rules_v3.json"
EXPECTED_RULE_IDS = {
    "D1",
    "D2",
    *(f"C{index}" for index in range(1, 15)),
    *(f"O{index}" for index in range(1, 15)),
    "O6a",
}


@lru_cache(maxsize=1)
def load_rules(path: str | Path | None = None) -> dict:
    rules_path = Path(path) if path else RULES_PATH
    if not rules_path.exists():
        raise FileNotFoundError(f"Official rules file not found: {rules_path}")
    rules = json.loads(rules_path.read_text(encoding="utf-8-sig"))
    validate_rules_integrity(rules)
    return rules


def all_rules(rules: dict | None = None) -> list[dict]:
    rules = rules or load_rules()
    return [
        *rules["fundamental_rules"],
        *rules["conops_rules"],
        *rules["opscon_rules"],
    ]


def validate_rules_integrity(rules: dict | None = None) -> bool:
    rules = rules or load_rules()
    if rules.get("version") != "3.0":
        raise ValueError("Rules version must be 3.0")
    identifiers = [rule.get("id") for rule in all_rules(rules)]
    missing = sorted(EXPECTED_RULE_IDS - set(identifiers))
    unexpected = sorted(set(identifiers) - EXPECTED_RULE_IDS)
    duplicates = sorted(
        identifier
        for identifier in set(identifiers)
        if identifiers.count(identifier) > 1
    )
    if missing:
        raise ValueError(f"Missing official rule IDs: {', '.join(missing)}")
    if unexpected:
        raise ValueError(
            f"Unexpected non-official rule IDs: {', '.join(unexpected)}"
        )
    if duplicates:
        raise ValueError(f"Duplicate official rule IDs: {', '.join(duplicates)}")
    for rule in all_rules(rules):
        missing_fields = [
            field
            for field in (
                "id",
                "name",
                "family",
                "type",
                "weight",
                "description",
                "expected_for_conops",
                "expected_for_opscon",
                "purpose",
                "penalty_logic",
            )
            if rule.get(field) in (None, "")
        ]
        if missing_fields:
            raise ValueError(
                f"Rule {rule.get('id', '?')} missing fields: "
                f"{', '.join(missing_fields)}"
            )
        if not (
            rule.get("expected")
            or rule.get("expected_for")
            or rule.get("forbidden_for")
        ):
            raise ValueError(
                f"Rule {rule['id']} has no expected status."
            )
    alpha = float(rules["scoring_policy"]["alpha"])
    beta = float(rules["scoring_policy"]["beta"])
    if alpha != 0.4:
        raise ValueError("Scoring alpha must be exactly 0.4")
    if beta != 1.5:
        raise ValueError("Scoring beta must be exactly 1.5")
    return True


def get_active_rules() -> list[dict]:
    return all_rules(load_rules())


def get_rule_by_id(rule_id: str) -> dict:
    normalized = rule_id.strip().upper()
    for rule in get_active_rules():
        if rule["id"].upper() == normalized:
            return rule
    raise KeyError(f"Unknown official rule ID: {rule_id}")


def get_rules_by_family(family: str) -> list[dict]:
    normalized = family.strip().lower()
    return [
        rule
        for rule in get_active_rules()
        if str(rule.get("family", "")).lower() == normalized
    ]


def build_rules_prompt_context(max_keywords: int = 6) -> str:
    rules = load_rules()
    lines = [
        f"Official ConOps/OpsCon rules version {rules['version']}.",
        "Use these rules as extraction guidance only; Python performs scoring.",
        "System-of-Systems concepts belong to OpsCon, not ConOps.",
        "Use stakeholder needs, expected outcomes, value criteria and value chain "
        "traceability only in their explicit meanings.",
    ]
    for rule in all_rules(rules):
        expected = rule.get("expected", rule.get("expected_for", []))
        keywords = ", ".join(rule.get("evidence_keywords", [])[:max_keywords])
        forbidden = ", ".join(rule.get("forbidden_elements", [])[:max_keywords])
        lines.append(
            f"{rule['id']} | {rule['name']} | expected={expected} | "
            f"weight={rule['weight']} | evidence={keywords or 'none'} | "
            f"forbidden={forbidden or 'none'}"
        )
    return "\n".join(lines)
