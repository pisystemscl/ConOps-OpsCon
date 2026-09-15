import json

from src.core.schema import ConOpsAnalysis
from src.llm.json_guard import (
    JsonGuardError,
    extract_json_from_response,
    repair_json_once,
    safe_llm_json_call,
    validate_json_schema,
)


class RepairingProvider:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, temperature=0.0):
        self.calls += 1
        if self.calls == 1:
            return "```json\n{document_title: 'Broken',}\n```"
        return json.dumps({"document_title": "Repaired"})


class InvalidSchemaProvider:
    def generate(self, prompt, temperature=0.0):
        return json.dumps({"requirements": [{"priority": "high"}]})


def test_json_guard_repairs_and_validates_json():
    repaired = repair_json_once("{document_title: \"Sample\",}")
    data = extract_json_from_response(repaired)

    assert data["document_title"] == "Sample"
    assert validate_json_schema(data, ConOpsAnalysis)


def test_safe_llm_json_call_uses_repair_attempts():
    result = safe_llm_json_call("prompt", ConOpsAnalysis, RepairingProvider())

    assert result.json_valid is True
    assert result.data["document_title"] in {"Broken", "Repaired"}
    assert result.json_repair_attempts >= 0


def test_safe_llm_json_call_reports_schema_error_details():
    try:
        safe_llm_json_call(
            "prompt",
            ConOpsAnalysis,
            InvalidSchemaProvider(),
            max_retries=1,
        )
    except JsonGuardError as error:
        assert "requirements.0.text" in str(error)
        assert error.raw_response
    else:
        raise AssertionError("Expected JsonGuardError")
