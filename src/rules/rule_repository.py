from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.domain.rules_loader import (
    all_rules,
    get_rule_by_id,
    get_rules_by_family,
    load_rules,
    validate_rules_integrity,
)


@dataclass(frozen=True)
class RulesRepository:
    """Repository for the official ConOps/OpsCon V3.0 rules."""

    path: str | Path | None = None

    @property
    def data(self) -> dict:
        return load_rules(self.path)

    @property
    def version(self) -> str:
        return str(self.data["version"])

    @property
    def source(self) -> str:
        return str(self.data.get("source", "Official Rules V3.0"))

    def validate(self) -> bool:
        return validate_rules_integrity(self.data)

    def all(self) -> list[dict]:
        return all_rules(self.data)

    def by_id(self, rule_id: str) -> dict:
        return get_rule_by_id(rule_id)

    def by_family(self, family: str) -> list[dict]:
        return get_rules_by_family(family)


def load_official_rules_repository(
    path: str | Path | None = None,
) -> RulesRepository:
    repository = RulesRepository(path)
    repository.validate()
    return repository
