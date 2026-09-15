from __future__ import annotations

import re
from difflib import SequenceMatcher

from src.core.schema import ConOpsAnalysis


BAD_REQUIREMENT_PATTERNS = (
    "shall support improve",
    "shall support support",
    "shall support the european atm network needs",
    "shall support in order to",
    "shall support this needs",
    "shall support it needs",
    "shall support the",
    "needs to be",
)

RISK_KEYWORDS = (
    "risk", "uncertainty", "shortage", "outage", "cyberattack",
    "cyber attack", "unavailability", "failure", "lack of", "depends on",
    "crisis", "disruption", "threat", "vulnerability", "constraint",
    "dependency", "certification issue", "fragmented",
)
MITIGATION_KEYWORDS = (
    "backup", "mitigation", "improves reliability", "enhanced resilience",
    "contingency solution", "redundancy", "fallback", "solution",
    "offer scalability", "allows to deal", "cyber-secure",
)
DRIVER_KEYWORDS = (
    "50000 flights", "50 000 flights", "needs to accommodate",
    "traffic demand", "performance target", "peak day",
)
REQUIREMENT_KEYWORDS = (
    "shall maintain", "shall support", "shall provide", "needs to",
    "must", "will ensure",
)

ACTOR_ALIASES = {
    "ansp": "ANSPs",
    "ansps": "ANSPs",
    "air navigation service provider": "ANSPs",
    "air navigation service providers": "ANSPs",
    "au": "Airspace Users",
    "aus": "Airspace Users",
    "airspace user": "Airspace Users",
    "network manager": "Network Manager",
    "nm": "Network Manager",
    "airport operator": "Airport Operators",
    "atc centre": "ATC centres",
    "atc center": "ATC centres",
    "atc centres": "ATC centres",
    "atc centers": "ATC centres",
    "military actor": "Military Actors",
    "military actors": "Military Actors",
    "oat": "Military Actors",
    "adsp": "ATM Data Service Providers",
    "adsps": "ATM Data Service Providers",
    "airport": "Airports",
    "airports": "Airports",
    "new entrant": "New Entrants",
    "new entrants": "New Entrants",
}


def normalize_actor_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name).strip()
    return ACTOR_ALIASES.get(cleaned.lower(), cleaned)


def clean_requirement_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    replacements = {
        "The system shall support improve": "The network shall enable improved",
        "The system shall support support": "The network shall support",
        "The system shall support the European ATM network needs to accommodate": (
            "The network shall accommodate"
        ),
        "The system shall support the European ATM Network will provide": (
            "The European ATM Network shall provide"
        ),
        "The network shall improve the European ATM Network needs to be substantially improved": (
            "The European ATM Network shall improve"
        ),
        "The system shall support a basis to ensure": (
            "The Network CONOPS shall ensure"
        ),
        "The system shall support in order to cope with this traffic demand,": (
            "The network shall improve"
        ),
        "The system shall support this needs to be achieved": (
            "The implementation shall be achieved"
        ),
        "The system shall support it needs to take into account": (
            "The implementation shall take into account"
        ),
    }
    for bad, good in replacements.items():
        text = re.sub(re.escape(bad), good, text, flags=re.IGNORECASE)
    return text


def requirement_quality_flags(text: str) -> list[str]:
    lowered = text.lower()
    flags = []
    if any(pattern in lowered for pattern in BAD_REQUIREMENT_PATTERNS):
        flags.append("BAD_REQUIREMENT_GRAMMAR")
    if len(text.split()) > 35:
        flags.append("TOO_LONG_REQUIREMENT")
    if not re.search(r"\b(shall|must|doit|devra)\b", text, re.IGNORECASE):
        flags.append("MISSING_NORMATIVE_VERB")
    return flags


def classify_risk_candidate(text: str) -> str:
    lowered = text.lower()
    if any(keyword in lowered for keyword in DRIVER_KEYWORDS):
        return "driver"
    if any(keyword in lowered for keyword in MITIGATION_KEYWORDS):
        return "mitigation"
    if any(keyword in lowered for keyword in REQUIREMENT_KEYWORDS):
        return "requirement"
    if any(keyword in lowered for keyword in RISK_KEYWORDS):
        return "risk"
    return "unknown"


def split_source_pages(source_text: str) -> list[tuple[int, str]]:
    matches = list(re.finditer(r"\[Page\s+(\d+)\]", source_text))
    if not matches:
        return []
    pages = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source_text)
        pages.append((int(match.group(1)), source_text[match.end():end].strip()))
    return pages


def find_page_for_evidence(evidence_quote: str, source_text: str) -> int | None:
    if not evidence_quote.strip():
        return None
    pages = split_source_pages(source_text)
    if not pages:
        return None
    normalized_quote = re.sub(r"\s+", " ", evidence_quote).lower().strip()
    probe = normalized_quote[:120]
    for page, text in pages:
        if probe and probe in re.sub(r"\s+", " ", text).lower():
            return page
    best_page = None
    best_score = 0.0
    for page, text in pages:
        normalized_text = re.sub(r"\s+", " ", text).lower()
        if len(normalized_text) > 2500:
            position = normalized_text.find(normalized_quote[:40])
            start = max(0, position - 500) if position >= 0 else 0
            normalized_text = normalized_text[start:start + 2500]
        score = SequenceMatcher(None, normalized_quote, normalized_text).ratio()
        if score > best_score:
            best_page, best_score = page, score
    return best_page if best_score >= 0.18 else None


def human_review_stats(analysis: ConOpsAnalysis) -> tuple[int, int, float]:
    reviewed = 0
    total = 0
    for stakeholder in analysis.stakeholders:
        total += 1
        reviewed += int(
            not stakeholder.evidence
            or stakeholder.evidence[0].page is None
        )
    for requirement in analysis.requirements:
        total += 1
        reviewed += int(
            bool(
                requirement_quality_flags(requirement.text)
                or requirement.quality_flags
            )
            or not requirement.evidence
            or requirement.evidence[0].page is None
        )
    for interface in analysis.interfaces:
        total += 1
        reviewed += int(
            not interface.source_actor
            or not interface.target_actor
            or not interface.exchanged_information
            or bool(interface.quality_flags)
            or not interface.evidence
            or interface.evidence[0].page is None
        )
    for risk in analysis.risks:
        total += 1
        reviewed += int(
            classify_risk_candidate(risk.description) != "risk"
            or not risk.evidence
            or risk.evidence[0].page is None
        )
    rate = reviewed / total if total else 1.0
    return reviewed, total, round(rate, 3)


def get_verdict(
    final_score: float,
    human_review_rate: float,
    classification_score: float,
) -> str:
    if (
        final_score >= 90
        and human_review_rate < 0.20
        and classification_score >= 0.85
    ):
        return "Excellent - candidat à validation experte"
    if final_score >= 80 and human_review_rate < 0.35:
        return "Bon - quelques corrections nécessaires"
    if final_score >= 65:
        return "Prometteur - revue humaine nécessaire"
    return "Insuffisant - extraction à reprendre"


def text_grounding_similarity(text: str, evidence: str) -> float:
    if not text.strip() or not evidence.strip():
        return 0.0
    return round(SequenceMatcher(None, text.lower(), evidence.lower()).ratio(), 3)


def score_entities(analysis: ConOpsAnalysis) -> float:
    score = 0.0
    stakeholder_count = len(analysis.stakeholders)
    service_count = len(analysis.expected_services)
    capability_count = len(analysis.capabilities) + len(analysis.enablers)
    system_count = len(analysis.existing_systems)
    score += 35 if stakeholder_count >= 6 else 25 if stakeholder_count >= 4 else 15 if stakeholder_count >= 2 else 5 if stakeholder_count else 0
    score += 25 if service_count >= 5 else 15 if service_count >= 3 else 7 if service_count else 0
    score += 25 if capability_count >= 5 else 15 if capability_count >= 3 else 7 if capability_count else 0
    score += 15 if system_count >= 2 else 8 if system_count == 1 else 0
    return min(score, 100.0)


def score_requirements(analysis: ConOpsAnalysis) -> float:
    if not analysis.requirements:
        return 0.0
    valid = 0
    for requirement in analysis.requirements:
        if (
            requirement.id
            and requirement.evidence
            and not requirement_quality_flags(requirement.text)
        ):
            valid += 1
    quality = 100 * valid / len(analysis.requirements)
    coverage = min(1.0, len(analysis.requirements) / 8)
    return round(quality * 0.75 + coverage * 25, 2)


def score_interfaces(analysis: ConOpsAnalysis) -> float:
    if not analysis.interfaces:
        return 0.0
    valid = sum(
        bool(interface.source and interface.target and interface.exchanged_information and interface.evidence)
        for interface in analysis.interfaces
    )
    count_score = min(len(analysis.interfaces) / 6, 1.0) * 50
    quality_score = valid / len(analysis.interfaces) * 50
    return round(count_score + quality_score, 2)


def score_risks(analysis: ConOpsAnalysis) -> float:
    if not analysis.risks:
        return 0.0
    valid = sum(
        classify_risk_candidate(risk.description) == "risk"
        and bool(risk.evidence)
        and bool(risk.mitigation)
        for risk in analysis.risks
    )
    coverage = min(len(analysis.risks) / 5, 1.0)
    return round(70 * valid / len(analysis.risks) + 30 * coverage, 2)


def classification_correctness(analysis: ConOpsAnalysis) -> float:
    total = len(analysis.risks) + len(analysis.requirements) + len(analysis.interfaces)
    if not total:
        return 0.0
    correct_risks = sum(
        classify_risk_candidate(risk.description) == "risk"
        for risk in analysis.risks
    )
    correct_requirements = sum(
        not requirement_quality_flags(requirement.text)
        for requirement in analysis.requirements
    )
    correct_interfaces = sum(
        bool(interface.source and interface.target and interface.exchanged_information)
        for interface in analysis.interfaces
    )
    return round(
        (correct_risks + correct_requirements + correct_interfaces) / total,
        3,
    )
