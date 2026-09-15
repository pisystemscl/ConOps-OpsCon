from __future__ import annotations
import re
from pathlib import Path
from src.core.schema import ConOpsAnalysis
from src.utils.io import slugify


def _id(value: str, prefix: str = "E") -> str:
    value = slugify(value, max_len=40)
    value = re.sub(r"[^a-zA-Z0-9_]", "_", value)
    if not value or value[0].isdigit():
        value = f"{prefix}_{value}"
    return value


def generate_sysml_v2(analysis: ConOpsAnalysis) -> str:
    pkg = _id(analysis.document_title, "ConOpsPackage")
    classification = analysis.document_classification or "Neither"
    lines: list[str] = [f"package {pkg} {{"]
    lines.append(
        "  // Generated as textual SysML v2 approximation from rule-based "
        "ConOps/OpsCon extraction. Validation is heuristic, not official."
    )
    lines.append(f"  // Rule-based classification: {classification}")
    lines.append("")
    lines.extend(
        [
            "  part def OperationalContext;",
            "  part def Stakeholder;",
            "  part def SystemElement;",
            "  part def StakeholderNeed;",
            "  part def ExpectedOutcome;",
            "  part def FutureService;",
            "  part def CapabilityGap;",
            "  part def FutureCapability;",
            "  part def FutureSolutionConstraint;",
            "  part def RequirementElement;",
            "  part def RiskElement;",
        ]
    )
    if classification in {"OpsCon", "Hybrid"}:
        lines.extend(
            [
                "  part def FutureOperationalSoS;",
                "  part def SoSBoundary;",
                "  part def ConstituentSystem;",
                "  part def OperationalService;",
                "  part def OperationalFlow;",
                "  part def InterfaceElement;",
                "  part def Governance;",
                "  part def OperationalDependency;",
                "  part def OperationalMode;",
            ]
        )
    lines.extend(["", "  part operationalContext : OperationalContext {"])
    for stakeholder in analysis.stakeholders:
        lines.append(
            f"    part {_id(stakeholder.name, 'stakeholder')} : Stakeholder; "
            f"// {stakeholder.role}"
        )
    for system in analysis.existing_systems:
        lines.append(f"    part {_id(system, 'system')} : SystemElement;")
    for service in analysis.expected_services:
        lines.append(
            f"    part {_id(service, 'service')} : FutureService;"
        )
    for capability in analysis.capabilities:
        lines.append(
            f"    part {_id(capability, 'capability')} : FutureCapability;"
        )
    for need in analysis.stakeholder_needs:
        lines.append(f"    part {_id(need, 'need')} : StakeholderNeed;")
    for outcome in analysis.expected_outcomes:
        lines.append(f"    part {_id(outcome, 'outcome')} : ExpectedOutcome;")
    for gap in analysis.capability_gaps:
        lines.append(f"    part {_id(gap, 'gap')} : CapabilityGap;")
    for constraint in analysis.constraints:
        lines.append(
            f"    part {_id(constraint, 'constraint')} : "
            "FutureSolutionConstraint;"
        )
    if classification in {"OpsCon", "Hybrid"}:
        lines.append(
            "    part futureOperationalSoS : FutureOperationalSoS;"
        )
        for system in analysis.constituent_systems:
            lines.append(
                f"    part {_id(system, 'constituent')} : ConstituentSystem;"
            )
    lines.extend(["  }", ""])

    for requirement in analysis.requirements:
        safe_text = requirement.text.replace('"', "'")
        evidence = requirement.evidence[0] if requirement.evidence else None
        trace = (
            f" source={evidence.source}, page={evidence.page}, "
            f"evidence={evidence.text[:160]}"
            if evidence
            else " evidence=missing"
        )
        lines.extend(
            [
                f"  requirement def {_id(requirement.id, 'REQ')} {{",
                f"    doc /* {safe_text} */",
                f"    //{trace}",
                "  }",
            ]
        )
    lines.append("")

    for index, interface in enumerate(analysis.interfaces, start=1):
        if classification not in {"OpsCon", "Hybrid"}:
            continue
        info = interface.exchanged_information.replace('"', "'")
        evidence = interface.evidence[0] if interface.evidence else None
        trace = (
            f"source={evidence.source}, page={evidence.page}, "
            f"evidence={evidence.text[:160]}"
            if evidence
            else "evidence=missing"
        )
        lines.extend(
            [
                f"  connection def Interface_{index:03d} {{",
                f"    end source : {_id(interface.source, 'src')};",
                f"    end target : {_id(interface.target, 'tgt')};",
                f"    doc /* exchanged_information: {info} */",
                f"    // {trace}",
                "  }",
            ]
        )
    for index, risk in enumerate(analysis.risks, start=1):
        evidence = risk.evidence[0] if risk.evidence else None
        trace = (
            f"source={evidence.source}, page={evidence.page}, "
            f"evidence={evidence.text[:160]}"
            if evidence
            else "evidence=missing"
        )
        lines.extend(
            [
                f"  part risk_{index:03d} : RiskElement;",
                f"  // risk: {risk.description[:180]}",
                f"  // {trace}",
            ]
        )
    if classification in {"OpsCon", "Hybrid"}:
        for flow in analysis.operational_flows:
            lines.append(f"  part {_id(flow, 'flow')} : OperationalFlow;")
        for item in analysis.governance:
            lines.append(f"  part {_id(item, 'governance')} : Governance;")
        for item in analysis.dependencies:
            lines.append(
                f"  part {_id(item, 'dependency')} : OperationalDependency;"
            )
        for item in analysis.operational_modes:
            lines.append(f"  part {_id(item, 'mode')} : OperationalMode;")
    lines.append("}")
    return "\n".join(lines)


def validate_sysml_text(
    sysml_text: str,
    analysis: ConOpsAnalysis | None = None,
) -> tuple[float, list[str]]:
    checks: list[tuple[bool, float, str]] = [
        (
            bool(re.search(r"^\s*package\s+[A-Za-z_]\w*\s*\{", sysml_text, re.MULTILINE)),
            0.10,
            "Declaration package invalide ou absente",
        ),
        (
            sysml_text.count("{") == sysml_text.count("}"),
            0.10,
            "Accolades non equilibrees",
        ),
        ("part def OperationalContext;" in sysml_text, 0.10, "OperationalContext absent"),
        (
            "part operationalContext : OperationalContext" in sysml_text,
            0.10,
            "Instance du contexte operationnel absente",
        ),
        (
            bool(re.search(r"\brequirement def\s+[A-Za-z_]\w*\s*\{", sysml_text)),
            0.15,
            "Aucune exigence SysML generee",
        ),
        (
            bool(re.search(r"\bpart def\s+[A-Za-z_]\w*\s*;", sysml_text)),
            0.10,
            "Aucune definition de part generee",
        ),
        ("doc /*" in sysml_text, 0.05, "Documentation des elements absente"),
        (
            "connection def" in sysml_text
            or (
                analysis is not None
                and analysis.document_classification not in {"OpsCon", "Hybrid"}
            ),
            0.10,
            "Aucune connexion/interface generee",
        ),
        (
            "Stakeholder" in sysml_text,
            0.05,
            "Aucun type Stakeholder genere",
        ),
        (
            "TODO" not in sysml_text and "undefined" not in sysml_text.lower(),
            0.05,
            "Placeholder TODO ou undefined detecte",
        ),
    ]
    if analysis is not None:
        requirement_count = len(re.findall(r"\brequirement def\b", sysml_text))
        checks.append(
            (
                requirement_count >= len(analysis.requirements),
                0.05,
                "Toutes les exigences JSON ne sont pas tracees dans SysML",
            )
        )
    errors = [message for passed, _, message in checks if not passed]
    score = sum(weight for passed, weight, _ in checks if passed)
    if analysis is not None:
        capability_terms = (
            "full dynamicity", "cross border", "cross-border", "ff-ice",
            "air-ground data exchange", "scalability", "resilience",
        )
        taxonomy_errors = [
            service
            for service in analysis.expected_services
            if any(term in service.lower() for term in capability_terms)
        ]
        if taxonomy_errors:
            score -= 0.10
            errors.append(
                "Capabilities ou enablers classes comme OperationalService"
            )
        if (
            analysis.document_classification in {"OpsCon", "Hybrid"}
            and len(analysis.interfaces) < 3
        ):
            score -= 0.10
            errors.append("Couverture des interfaces insuffisante pour le modele")
        if len(analysis.stakeholders) < 3:
            score -= 0.05
            errors.append("Couverture des stakeholders insuffisante pour le modele")
    return round(max(0.0, score), 3), errors


def save_sysml(path: str | Path, analysis: ConOpsAnalysis) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_sysml_v2(analysis), encoding="utf-8")
    return path
