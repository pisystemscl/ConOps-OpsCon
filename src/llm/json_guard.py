from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from src.llm.json_repair import repair_json, safe_extract_json


@dataclass
class JsonGuardResult:
    data: dict[str, Any]
    json_valid: bool
    json_repair_attempts: int = 0
    json_error_message: str = ""
    raw_response: str = ""
    repaired_response: str = ""
    attempts: list[str] = field(default_factory=list)


class JsonGuardError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        raw_response: str = "",
        repaired_response: str = "",
        attempts: list[str] | None = None,
    ):
        super().__init__(message)
        self.raw_response = raw_response
        self.repaired_response = repaired_response
        self.attempts = attempts or []


def extract_json_from_response(raw_text: str) -> dict[str, Any]:
    return safe_extract_json(raw_text)


def _validation_error_message(error: Exception) -> str:
    if isinstance(error, ValidationError):
        details = []
        for item in error.errors()[:8]:
            location = ".".join(str(part) for part in item.get("loc", ()))
            message = item.get("msg", "validation error")
            details.append(f"{location or '<root>'}: {message}")
        suffix = ""
        if len(error.errors()) > len(details):
            suffix = f" (+{len(error.errors()) - len(details)} autre(s))"
        return "; ".join(details) + suffix
    return str(error)


def assert_json_schema(data: dict[str, Any], schema) -> None:
    if schema is None:
        if not isinstance(data, dict):
            raise TypeError("Expected a JSON object.")
        return
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema.model_validate(data)
    elif isinstance(schema, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing required keys: {', '.join(missing)}")
    elif callable(schema):
        schema(data)


def validate_json_schema(data: dict[str, Any], schema) -> bool:
    try:
        assert_json_schema(data, schema)
        return True
    except (ValidationError, ValueError, TypeError):
        return False


def repair_json_once(raw_text: str) -> str:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    candidate = raw_text[start : end + 1] if start != -1 and end > start else raw_text
    return repair_json(candidate)


def repair_json_with_llm(raw_text: str, schema, provider) -> str:
    schema_hint = ""
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema_hint = json.dumps(schema.model_json_schema(), ensure_ascii=False)[:6000]
    elif isinstance(schema, dict):
        schema_hint = json.dumps(schema, ensure_ascii=False)[:6000]
    prompt = (
        "JSON repair. Result shape: one valid JSON object, "
        "without Markdown and without additional facts.\n\nSchema hint:\n"
        f"{schema_hint}\n\nInvalid JSON:\n{raw_text[:12000]}"
    )
    if hasattr(provider, "generate"):
        return provider.generate(prompt, temperature=0.0)
    raise ValueError("Provider interface missing generate(prompt, temperature=0.0).")


def safe_llm_json_call(
    prompt: str,
    schema,
    provider,
    max_retries: int = 3,
) -> JsonGuardResult:
    raw = provider.generate(prompt, temperature=0.0)
    attempts = [raw]
    last_error = ""

    for attempt_index in range(max_retries):
        candidate = attempts[-1]
        try:
            data = extract_json_from_response(candidate)
            try:
                assert_json_schema(data, schema)
                return JsonGuardResult(
                    data=data,
                    json_valid=True,
                    json_repair_attempts=attempt_index,
                    raw_response=raw,
                    repaired_response=candidate if attempt_index else "",
                    attempts=attempts,
                )
            except (ValidationError, ValueError, TypeError) as schema_error:
                last_error = (
                    "JSON parsed but schema validation failed: "
                    f"{_validation_error_message(schema_error)}"
                )
        except Exception as exc:
            last_error = str(exc)

        if attempt_index == 0:
            attempts.append(repair_json_once(candidate))
        else:
            attempts.append(repair_json_with_llm(candidate, schema, provider))

    raise JsonGuardError(
        "JSON invalide apres "
        f"{max_retries} tentative(s) de reparation: {last_error}",
        raw_response=raw,
        repaired_response=attempts[-1] if attempts else "",
        attempts=attempts,
    )
