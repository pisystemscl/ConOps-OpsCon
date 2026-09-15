from __future__ import annotations

import ast
import json
import re
from typing import Any


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    return list(dict.fromkeys(candidates))


def repair_json(candidate: str) -> str:
    """Apply conservative repairs to common malformed LLM JSON."""
    repaired = candidate.lstrip("\ufeff").strip()
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(
        r"([{,]\s*)([A-Za-z_][\w-]*)(\s*:)",
        r'\1"\2"\3',
        repaired,
    )
    repaired = re.sub(
        r'([}\]"])[\s\r\n]+(?="[^"\r\n]+"\s*:)',
        r"\1,",
        repaired,
    )
    return repaired.replace("\u201c", '"').replace("\u201d", '"')


def safe_extract_json(text: str) -> dict[str, Any]:
    """Extract and conservatively repair a JSON object from an LLM answer."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    last_error: Exception | None = None
    for candidate in _json_candidates(text):
        for variant in (candidate, repair_json(candidate)):
            try:
                data = json.loads(variant)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError as exc:
                last_error = exc
        try:
            data = ast.literal_eval(candidate)
            if isinstance(data, dict):
                return data
        except (ValueError, SyntaxError) as exc:
            last_error = exc

    detail = f" ({last_error})" if last_error else ""
    raise ValueError(
        f"Aucun JSON valide trouve dans la reponse du modele{detail}."
    )
