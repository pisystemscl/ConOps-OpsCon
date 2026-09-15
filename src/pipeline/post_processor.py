from __future__ import annotations
import re
from src.core.schema import (
    ConOpsAnalysis,
    Evidence,
    Interface,
    Requirement,
    Risk,
    Stakeholder,
)
from src.pipeline.quality import (
    classify_risk_candidate,
    clean_requirement_text,
    find_page_for_evidence,
    normalize_actor_name,
    requirement_quality_flags,
)
from src.postprocessing.requirement_cleaner import clean_requirement

TOC_PATTERNS = [
    r"^\s*(table of contents|contents|list of figures|list of tables|document history|change record)\s*$",
    r"^\s*(figure|table)\s+\d+[:\.]",
    r"^\s*\d+(?:\.\d+)*\s+[A-Z0-9 /&()\-,:]{8,}\s*\.{3,}\s*\d+\s*$",
    r"^\s*\d+(?:\.\d+)*\s+[A-Z][A-Z0-9 /&()\-,:]{12,}\s*$",
    r".*\.{6,}.*",
]

BAD_PLACEHOLDERS = {
    "acteur/système source", "acteur/système cible", "source", "target", "à définir", "unknown", "n/a"
}


def is_toc_or_noise(text: str) -> bool:
    if not text:
        return True
    t = re.sub(r"\s+", " ", str(text)).strip()
    if len(t) < 10:
        return True
    low = t.lower()
    if any(p in low for p in ["copyright", "effective date", "check the", "page "]):
        return True
    for pat in TOC_PATTERNS:
        if re.match(pat, t, flags=re.IGNORECASE):
            return True
    letters = re.findall(r"[A-Za-zÀ-ÿ]", t)
    if letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio > 0.82 and len(t) > 25:
            return True
    return False


def clean_text_item(text: str, max_len: int = 360) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"\s*\.{3,}\s*", " — ", text)
    return text[:max_len].strip()


def dedupe_clean(items: list[str], max_items: int = 10) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        cleaned = clean_text_item(item)
        if is_toc_or_noise(cleaned):
            continue
        key = cleaned.lower()[:160]
        if key not in seen:
            out.append(cleaned)
            seen.add(key)
        if len(out) >= max_items:
            break
    return out


def extract_sentences(source_text: str) -> list[str]:
    text = re.sub(r"\[Page \d+\]", " ", source_text)
    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [clean_text_item(s, 500) for s in sentences if 45 <= len(s.strip()) <= 500 and not is_toc_or_noise(s)]


def find_sentences(source_text: str, keywords: list[str], limit: int = 6) -> list[str]:
    out = []
    for s in extract_sentences(source_text):
        low = s.lower()
        if any(k.lower() in low for k in keywords):
            out.append(s)
        if len(out) >= limit:
            break
    return out


def evidence_from_sentence(
    sentence: str,
    source_text: str = "",
) -> list[Evidence]:
    return [
        Evidence(
            text=sentence[:280],
            source="document_utilisateur",
            page=find_page_for_evidence(sentence, source_text),
        )
    ]


def _enrich_stakeholders(
    analysis: ConOpsAnalysis,
    source_text: str,
) -> None:
    stakeholder_terms = {
        "Airspace Users": ["airspace users", "AUs"],
        "Airport Operators": ["airport operators"],
        "ANSPs": ["ANSPs", "air navigation service providers"],
        "Network Manager": ["Network Manager", "NMOC"],
        "Operational Stakeholders": ["operational stakeholders"],
        "Network Actors": ["network actors"],
        "ATC centres": ["ATC centres", "ATC centers"],
        "Military Actors": ["military actors", "OAT flight plans", "OAT"],
        "ATM Data Service Providers": ["ADSPs", "ATM data service providers"],
        "Airports": ["airports"],
        "New Entrants": ["new entrants", "drones", "UAV"],
    }
    existing = {normalize_actor_name(item.name).lower() for item in analysis.stakeholders}
    source_lower = source_text.lower()
    for canonical, aliases in stakeholder_terms.items():
        if canonical.lower() in existing:
            continue
        matched_alias = next(
            (alias for alias in aliases if alias.lower() in source_lower),
            None,
        )
        if not matched_alias:
            continue
        sentences = find_sentences(source_text, [matched_alias], limit=1)
        evidence = (
            evidence_from_sentence(sentences[0], source_text)
            if sentences
            else []
        )
        analysis.stakeholders.append(
            Stakeholder(
                name=canonical,
                role="Acteur opérationnel identifié dans le ConOps",
                interest="Participer au concept d'opérations et à ses échanges",
                influence_level="medium",
                evidence=evidence,
            )
        )
        existing.add(canonical.lower())


def _enrich_interfaces(
    analysis: ConOpsAnalysis,
    source_text: str,
) -> None:
    rules = [
        (
            ["ff-ice", "efpl", "4d business", "mission trajectories"],
            "Airspace Users",
            "Network / ATFCM",
            "4D business/mission trajectories via FF-ICE/eFPL",
        ),
        (
            ["aop/nop", "airport operations plan", "network operations plan"],
            "Airports",
            "Network Manager",
            "AOP/NOP planning and real-time operational data",
        ),
        (
            ["oat flight", "military atm demand"],
            "Military Actors",
            "Network / ATFCM",
            "OAT flight plans and military ATM demand",
        ),
        (
            ["swim"],
            "Network Actors",
            "Network Manager",
            "SWIM air-ground, ground-ground and civil-military information",
        ),
        (
            ["real-time data", "real time data", "tactical updates"],
            "ATC centres",
            "Network Manager",
            "Real-time traffic information and tactical updates",
        ),
    ]
    existing = {
        (
            normalize_actor_name(item.source).lower(),
            normalize_actor_name(item.target).lower(),
            item.exchanged_information.lower(),
        )
        for item in analysis.interfaces
    }
    source_lower = source_text.lower()
    for keywords, source, target, information in rules:
        matched = [keyword for keyword in keywords if keyword in source_lower]
        if not matched:
            continue
        sentences = find_sentences(source_text, matched, limit=1)
        if not sentences:
            continue
        key = (source.lower(), target.lower(), information.lower())
        if key in existing:
            continue
        analysis.interfaces.append(
            Interface(
                source=source,
                target=target,
                exchanged_information=information,
                criticality="high",
                evidence=evidence_from_sentence(sentences[0], source_text),
            )
        )
        existing.add(key)


def _reclassify_services(analysis: ConOpsAnalysis, source_text: str) -> None:
    capability_terms = (
        "full dynamicity", "cross border", "cross-border",
        "air-ground data exchange", "ff-ice", "scalability", "resilience",
    )
    enabler_terms = (
        "cloud", "cyber-secure", "data service provider", "adsp",
    )
    services = []
    for service in analysis.expected_services:
        lowered = service.lower()
        if any(term in lowered for term in enabler_terms):
            analysis.enablers.append(service)
        elif any(term in lowered for term in capability_terms):
            analysis.capabilities.append(service)
        else:
            services.append(service)
    operational_services = [
        "Optimised Network Design and Utilisation",
        "Optimum Capacity and Flight Efficiency Planning",
        "Trajectory and Cooperative Traffic Management",
        "Airport and TMA - Network Integration",
        "Network Components/system and CNS infrastructure evolutions",
    ]
    source_lower = source_text.lower()
    for service in operational_services:
        if service.lower() in source_lower:
            services.append(service)
    analysis.expected_services = dedupe_clean(services, max_items=10)
    analysis.capabilities = dedupe_clean(analysis.capabilities, max_items=12)
    analysis.enablers = dedupe_clean(analysis.enablers, max_items=10)


def normalize_requirement_text(text: str) -> str:
    text = clean_text_item(text, 450)
    if re.search(r"\b(shall|must|doit|devra)\b", text, flags=re.IGNORECASE):
        return text
    text = re.sub(
        r"^(further|therefore|moreover|in addition|additionally)[,:\s]+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(it|this document|the conops)\s+(provides|defines|describes)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.rstrip(" .;:")
    if not text:
        return ""
    return clean_requirement_text(
        f"The system shall support {text[0].lower() + text[1:]}."
    )


def enrich_analysis(analysis: ConOpsAnalysis, source_text: str) -> ConOpsAnalysis:
    for field in [
        "as_is", "to_be", "existing_systems", "existing_services", "expected_services",
        "gaps", "capabilities", "constraints", "assumptions", "things_to_change", "things_to_avoid"
    ]:
        setattr(analysis, field, dedupe_clean(getattr(analysis, field), max_items=10))

    stakeholders = []
    seen = set()
    for st in analysis.stakeholders:
        name = normalize_actor_name(clean_text_item(st.name, 80))
        if is_toc_or_noise(name):
            continue
        key = name.lower()
        if key not in seen:
            st.name = name
            stakeholders.append(st)
            seen.add(key)
    analysis.stakeholders = stakeholders[:15]
    _enrich_stakeholders(analysis, source_text)

    interfaces = []
    for interface_index, itf in enumerate(analysis.interfaces, start=1):
        src = normalize_actor_name(clean_text_item(itf.source, 120))
        tgt = normalize_actor_name(clean_text_item(itf.target, 120))
        exch = clean_text_item(itf.exchanged_information, 300)
        if src.lower() in BAD_PLACEHOLDERS or tgt.lower() in BAD_PLACEHOLDERS:
            continue
        if is_toc_or_noise(exch):
            continue
        itf.source = itf.source_actor = src
        itf.target = itf.target_actor = tgt
        itf.exchanged_information = itf.exchanged_item = exch
        itf.interface_id = itf.interface_id or f"INT-{interface_index:03d}"
        itf.name = itf.name or exch
        itf.direction = itf.direction or f"{src} -> {tgt}"
        itf.flow_type = (
            itf.flow_type
            if itf.flow_type in {"Information", "Material", "Energy", "Hybrid"}
            else "Information"
        )
        flags = list(itf.quality_flags)
        if not src or not tgt:
            flags.append("MISSING_INTERFACE_ACTOR")
        if not exch:
            flags.append("MISSING_EXCHANGED_ITEM")
        if not itf.evidence:
            flags.append("MISSING_EVIDENCE")
        elif itf.evidence[0].page is None:
            flags.append("MISSING_EVIDENCE_PAGE")
        itf.quality_flags = list(dict.fromkeys(flags))
        interfaces.append(itf)
    analysis.interfaces = interfaces[:12]
    _enrich_interfaces(analysis, source_text)
    for interface_index, itf in enumerate(analysis.interfaces, start=1):
        itf.interface_id = itf.interface_id or f"INT-{interface_index:03d}"
        itf.source_actor = itf.source_actor or itf.source
        itf.target_actor = itf.target_actor or itf.target
        itf.exchanged_item = itf.exchanged_item or itf.exchanged_information
        itf.name = itf.name or itf.exchanged_item
        itf.direction = itf.direction or (
            f"{itf.source_actor} -> {itf.target_actor}"
        )
        flags = list(itf.quality_flags)
        if not itf.source_actor or not itf.target_actor:
            flags.append("MISSING_INTERFACE_ACTOR")
        if not itf.evidence:
            flags.append("MISSING_EVIDENCE")
        elif itf.evidence[0].page is None:
            flags.append("MISSING_EVIDENCE_PAGE")
        itf.quality_flags = list(dict.fromkeys(flags))
    _reclassify_services(analysis, source_text)

    if len(analysis.gaps) < 2:
        analysis.gaps = dedupe_clean(
            analysis.gaps + find_sentences(source_text, ["gap", "shortcoming", "challenge", "barrier", "constraint", "problem", "limitation", "not covered", "missing", "lack"], 8),
            max_items=8,
        )

    existing_req_texts = {r.text.lower() for r in analysis.requirements}
    req_candidates = find_sentences(source_text, ["shall", "must", "requirement", "required", "needs to", "need to", "will need", "is required"], 12)
    next_id = len(analysis.requirements) + 1
    for s in req_candidates:
        if len(analysis.requirements) >= 8:
            break
        if s.lower() in existing_req_texts:
            continue
        analysis.requirements.append(Requirement(
            id=f"REQ-{next_id:03d}",
            text=s,
            priority="medium",
            source_category="extracted_high_level",
            evidence=evidence_from_sentence(s, source_text),
        ))
        next_id += 1

    for s in (analysis.expected_services + analysis.capabilities)[:6]:
        if len(analysis.requirements) >= 6:
            break
        txt = f"Le futur systeme doit permettre ou soutenir : {s}"
        analysis.requirements.append(Requirement(
            id=f"REQ-{next_id:03d}", text=txt, priority="medium", source_category="inferred_from_service", evidence=[]
        ))
        next_id += 1

    cleaned_reqs = []
    seen_req = set()
    for i, req in enumerate(analysis.requirements, start=1):
        source_item = clean_text_item(req.text, 450)
        txt = clean_requirement_text(normalize_requirement_text(source_item))
        if is_toc_or_noise(txt):
            continue
        if not req.evidence and source_item.lower() in source_text.lower():
            req.evidence = evidence_from_sentence(source_item, source_text)
        key = txt.lower()[:180]
        if key in seen_req:
            continue
        req.id = req.id or f"REQ-{i:03d}"
        req.text = txt
        req = clean_requirement(req)
        flags = list(
            dict.fromkeys(
                requirement_quality_flags(req.text) + req.quality_flags
            )
        )
        req.quality_flags = flags
        if flags:
            analysis.raw_notes.setdefault("quality_flags", []).append(
                {"item_id": req.id, "flags": flags}
            )
        cleaned_reqs.append(req)
        seen_req.add(key)
    analysis.requirements = cleaned_reqs[:10]

    risk_candidates = find_sentences(
        source_text,
        [
            "risk", "uncertainty", "shortage", "unavailability", "outage",
            "cyberattack", "cyber attack", "failure", "constraint",
            "dependency", "certification", "lack of", "fragmentation",
            "fragmented", "crisis", "disruption",
        ],
        14,
    )
    risk_texts = {r.description.lower() for r in analysis.risks}
    for s in risk_candidates:
        if len(analysis.risks) >= 6:
            break
        if s.lower() in risk_texts:
            continue
        analysis.risks.append(Risk(
            description=s,
            impact="medium",
            probability="medium",
            mitigation="A preciser par validation experte et architecture cible",
            evidence=evidence_from_sentence(s, source_text),
        ))

    cleaned_risks = []
    seen_risk = set()
    for risk in analysis.risks:
        desc = clean_text_item(risk.description, 450)
        if is_toc_or_noise(desc):
            continue
        category = classify_risk_candidate(desc)
        if category == "driver":
            analysis.drivers.append(desc)
            continue
        if category == "mitigation":
            analysis.mitigations.append(desc)
            continue
        if category == "requirement":
            analysis.constraints.append(desc)
            continue
        if category == "unknown":
            analysis.constraints.append(desc)
            continue
        key = desc.lower()[:180]
        if key not in seen_risk:
            risk.description = desc
            cleaned_risks.append(risk)
            seen_risk.add(key)
    analysis.risks = cleaned_risks[:8]
    analysis.drivers = dedupe_clean(analysis.drivers, max_items=10)
    analysis.mitigations = dedupe_clean(analysis.mitigations, max_items=10)
    analysis.constraints = dedupe_clean(analysis.constraints, max_items=12)

    traceable_items = (
        list(analysis.stakeholders)
        + list(analysis.requirements)
        + list(analysis.interfaces)
        + list(analysis.risks)
    )
    for item in traceable_items:
        for evidence in item.evidence:
            if evidence.text and evidence.page is None:
                evidence.page = find_page_for_evidence(evidence.text, source_text)

    if not analysis.future_actions:
        from src.core.schema import ActionItem
        analysis.future_actions = [
            ActionItem(action="Validation des stakeholders et de leurs responsabilites avec les experts metier", owner="Equipe MBSE", horizon="court terme", rationale="Consolidation de la qualite d'extraction"),
            ActionItem(action="Transformation des services et gaps valides en exigences haut niveau", owner="Equipe systemes", horizon="moyen terme", rationale="Preparation de la modelisation SysML v2"),
            ActionItem(action="Comparaison des configurations LLM/RAG/fine-tuning sur plusieurs ConOps", owner="Equipe systemes", horizon="moyen terme", rationale="Evaluation experimentale documentee"),
        ]

    if not analysis.current_situation:
        analysis.current_situation = list(analysis.as_is)
    if not analysis.change_drivers:
        analysis.change_drivers = list(analysis.drivers or analysis.gaps)
    if not analysis.future_services:
        analysis.future_services = list(analysis.expected_services)
    if not analysis.capability_gaps:
        analysis.capability_gaps = list(analysis.gaps)
    if not analysis.future_capabilities:
        analysis.future_capabilities = list(analysis.capabilities)
    analysis.raw_notes["stakeholder_needs"] = list(analysis.stakeholder_needs)
    analysis.raw_notes["service_allocations"] = [
        value
        for value in analysis.opscon_elements
        if "service" in value.lower() and "allocat" in value.lower()
    ]
    analysis.raw_notes["operational_flows"] = list(analysis.operational_flows)
    analysis.raw_notes["constituent_systems"] = list(
        analysis.constituent_systems
    )
    analysis.raw_notes["governance"] = list(analysis.governance)
    analysis.raw_notes["dependencies"] = list(analysis.dependencies)
    analysis.raw_notes["operational_modes"] = list(analysis.operational_modes)

    return analysis
