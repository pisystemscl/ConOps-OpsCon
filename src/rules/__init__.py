"""Compatibility facade for rule-loading helpers.

The domain implementation lives in :mod:`src.domain.rules_loader`.
"""

from src.domain.rules_loader import (
    build_rules_prompt_context,
    get_active_rules,
    get_rule_by_id,
    get_rules_by_family,
    load_rules,
    validate_rules_integrity,
)

__all__ = [
    "build_rules_prompt_context",
    "get_active_rules",
    "get_rule_by_id",
    "get_rules_by_family",
    "load_rules",
    "validate_rules_integrity",
]
