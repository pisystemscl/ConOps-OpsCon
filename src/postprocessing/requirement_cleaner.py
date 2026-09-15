from __future__ import annotations

import re

from src.core.schema import Requirement


ACTION_VERBS = (
    "allow",
    "collect",
    "communicate",
    "detect",
    "enable",
    "exchange",
    "maintain",
    "manage",
    "monitor",
    "prevent",
    "process",
    "provide",
    "record",
    "reduce",
    "share",
    "support",
    "validate",
)

DESCRIPTIVE_PREFIXES = (
    "the difference is that",
    "this document describes",
    "the concept explains",
    "it is important to note",
    "in order to",
)


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip(" .;:")


def detect_bad_requirement_grammar(text: str) -> bool:
    return not bool(re.match(r"^The system shall\s+[A-Za-z][A-Za-z-]+", _compact(text)))


def detect_too_long_requirement(text: str, *, max_words: int = 30) -> bool:
    return len(_compact(text).split()) > max_words


def detect_missing_action_verb(text: str) -> bool:
    return not any(
        re.search(rf"\b{re.escape(verb)}\b", text, re.IGNORECASE)
        for verb in ACTION_VERBS
    )


def detect_copied_evidence_as_requirement(
    text: str,
    evidence_quote: str,
) -> bool:
    cleaned = _compact(text).lower()
    evidence = _compact(evidence_quote).lower()
    return bool(evidence and (cleaned in evidence or evidence in cleaned))


def generate_requirement_from_evidence(evidence_quote: str) -> str:
    body = _compact(evidence_quote)
    body = re.sub(
        r"^(the\s+\w+(?:\s+\w+){0,3})\s+(shall|must|should)\s+",
        "",
        body,
        flags=re.IGNORECASE,
    )
    if detect_missing_action_verb(body):
        body = f"support {body[0].lower() + body[1:]}" if body else "support the validated operational need"
    words = body.split()[:27]
    return "The system shall " + " ".join(words).rstrip(" .;:") + "."


def clean_requirement_text(raw_text: str, evidence_quote: str = "") -> tuple[str, list[str], float]:
    requirement = Requirement(
        id="REQ-TEMP",
        text=raw_text or generate_requirement_from_evidence(evidence_quote),
        evidence=[{"text": evidence_quote}] if evidence_quote else [],
    )
    cleaned = clean_requirement(requirement)
    flags = list(cleaned.quality_flags)
    if detect_copied_evidence_as_requirement(cleaned.text, evidence_quote):
        flags.append("COPIED_EVIDENCE_AS_REQUIREMENT")
    if detect_missing_action_verb(cleaned.text):
        flags.append("MISSING_ACTION_VERB")
    return (
        cleaned.text,
        list(dict.fromkeys(flags)),
        float(cleaned.reformulation_confidence or 0.0),
    )


def score_requirement_quality(requirement: Requirement) -> float:
    flags = set(requirement.quality_flags)
    score = 100.0
    penalties = {
        "BAD_REQUIREMENT_GRAMMAR": 25,
        "TOO_LONG_REQUIREMENT": 15,
        "MISSING_ACTION_VERB": 20,
        "COPIED_EVIDENCE_AS_REQUIREMENT": 15,
        "MISSING_EVIDENCE": 20,
        "LOW_CONFIDENCE_REFORMULATION": 10,
    }
    for flag, penalty in penalties.items():
        if flag in flags:
            score -= penalty
    return round(max(0.0, score), 2)


def clean_requirement(
    requirement: Requirement,
    *,
    max_words: int = 30,
) -> Requirement:
    source_text = _compact(requirement.text)
    lowered = source_text.lower()
    confidence = 1.0

    if lowered.startswith("the system shall"):
        body = _compact(source_text[len("the system shall"):])
    else:
        body = re.sub(
            r"^(the\s+\w+(?:\s+\w+){0,3})\s+(shall|must)\s+",
            "",
            source_text,
            flags=re.IGNORECASE,
        )
        body = re.sub(
            r"^(shall|must|should|needs? to|required to)\s+",
            "",
            body,
            flags=re.IGNORECASE,
        )
        confidence = 0.75

    body_lower = body.lower()
    for prefix in DESCRIPTIVE_PREFIXES:
        if body_lower.startswith(prefix):
            body = _compact(body[len(prefix):])
            confidence = min(confidence, 0.55)
            break

    if not any(
        re.search(rf"\b{re.escape(verb)}\b", body, re.IGNORECASE)
        for verb in ACTION_VERBS
    ):
        body = f"support {body[0].lower() + body[1:]}" if body else "support the stated operational need"
        confidence = min(confidence, 0.6)

    words = body.split()
    was_too_long = len(words) + 3 > max_words
    if was_too_long:
        body = " ".join(words[: max_words - 3]).rstrip(",;:")
        confidence = min(confidence, 0.65)

    cleaned = f"The system shall {body}."
    flags: list[str] = []
    if not re.match(
        r"^The system shall\s+[A-Za-z][A-Za-z-]+",
        cleaned,
    ):
        flags.append("BAD_REQUIREMENT_GRAMMAR")
    if was_too_long:
        flags.append("TOO_LONG_REQUIREMENT")
    if not requirement.evidence or not requirement.evidence[0].text.strip():
        flags.append("MISSING_EVIDENCE")
    if not requirement.evidence or requirement.evidence[0].page is None:
        flags.append("MISSING_EVIDENCE_PAGE")
    if confidence < 0.7:
        flags.append("LOW_CONFIDENCE_REFORMULATION")
    if detect_missing_action_verb(cleaned):
        flags.append("MISSING_ACTION_VERB")

    requirement.text = cleaned
    requirement.quality_flags = list(dict.fromkeys(flags))
    requirement.reformulation_confidence = confidence
    return requirement


def clean_requirements(
    requirements: list[Requirement],
    *,
    max_words: int = 30,
) -> list[Requirement]:
    return [
        clean_requirement(requirement, max_words=max_words)
        for requirement in requirements
    ]
