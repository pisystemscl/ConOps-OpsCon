from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.schema import Evidence, Interface


FLOW_KEYWORDS = {
    "Information": ("data", "information", "message", "api", "communication", "exchange"),
    "Material": ("material", "vehicle", "passenger", "cargo", "asset"),
    "Energy": ("energy", "power", "fuel", "charging", "electric"),
}


@dataclass(frozen=True)
class InterfaceExtractionResult:
    interfaces: list[Interface]
    interface_completeness_score: float


def _flow_type(text: str) -> str:
    lowered = text.lower()
    matched = [
        flow for flow, terms in FLOW_KEYWORDS.items()
        if any(term in lowered for term in terms)
    ]
    if len(matched) > 1:
        return "Hybrid"
    return matched[0] if matched else "Information"


def _page_for(text: str, source_text: str) -> int | None:
    current_page = None
    for line in source_text.splitlines():
        match = re.match(r"\[Page\s+(\d+)\]", line.strip(), re.I)
        if match:
            current_page = int(match.group(1))
        if text[:80].lower() in line.lower():
            return current_page
    return None


def extract_structured_interfaces(
    source_text: str,
    *,
    classification: str = "ConOps",
    limit: int = 6,
) -> InterfaceExtractionResult:
    patterns = (
        r"(?P<src>[A-Z][A-Za-z0-9 /\-&]{2,40})\s+(?:exchanges?|shares?|provides?|sends?)\s+(?P<item>.{8,100}?)\s+(?:with|to)\s+(?P<tgt>[A-Z][A-Za-z0-9 /\-&]{2,40})",
        r"(?P<src>[A-Z][A-Za-z0-9 /\-&]{2,40})\s+(?:is integrated with|interfaces with|connects to)\s+(?P<tgt>[A-Z][A-Za-z0-9 /\-&]{2,40})",
    )
    candidates: list[Interface] = []
    compact = re.sub(r"\s+", " ", source_text)
    compact = re.sub(r"\[Page\s+\d+\]\s*", "", compact, flags=re.I)
    for pattern in patterns:
        for match in re.finditer(pattern, compact):
            src = match.groupdict().get("src", "").strip(" .,:;")
            tgt = match.groupdict().get("tgt", "").strip(" .,:;")
            item = match.groupdict().get("item", "operational information").strip(" .,:;")
            quote = match.group(0).strip()
            interface = Interface(
                interface_id=f"INT-{len(candidates) + 1:03d}",
                name=item[:80],
                source_actor=src,
                target_actor=tgt,
                exchanged_item=item,
                flow_type=_flow_type(quote),
                direction=f"{src} -> {tgt}" if src and tgt else "",
                trigger="",
                related_rule_id="O8",
                evidence_quote=quote,
                source_page=_page_for(quote, source_text),
                confidence=1.0 if src and tgt and item else 0.45,
                human_review_needed=not (src and tgt and item),
                evidence=[
                    Evidence(text=quote, page=_page_for(quote, source_text))
                ],
            )
            if not interface.source_actor or not interface.target_actor:
                interface.quality_flags.append("MISSING_INTERFACE_ACTOR")
            if not interface.flow_type:
                interface.quality_flags.append("MISSING_FLOW_TYPE")
            if classification == "ConOps":
                interface.quality_flags.append("POSSIBLE_OPSCON_DRIFT")
            candidates.append(interface)
            if len(candidates) >= limit:
                break
        if len(candidates) >= limit:
            break
    complete = sum(
        bool(item.source_actor and item.target_actor and item.flow_type)
        for item in candidates
    )
    score = round(complete / len(candidates), 3) if candidates else 0.0
    return InterfaceExtractionResult(candidates, score)
