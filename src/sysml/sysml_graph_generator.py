from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyvis.network import Network

from src.core.schema import ConOpsAnalysis
from src.pipeline.sysml_generator import generate_sysml_v2, save_sysml


def sysml_traceability_rows(analysis: ConOpsAnalysis) -> list[dict]:
    rows: list[dict] = []
    for requirement in analysis.requirements:
        evidence = requirement.evidence[0] if requirement.evidence else None
        rows.append(
            {
                "source_evidence": evidence.text if evidence else "",
                "source_page": evidence.page if evidence else None,
                "extracted_item": requirement.id,
                "item_type": "Requirement",
                "sysml_element": requirement.id,
                "rule_id": "C5" if requirement.source_category == "ConOps" else "O8",
                "confidence": requirement.reformulation_confidence or 0.7,
            }
        )
    for interface in analysis.interfaces:
        evidence = interface.evidence[0] if interface.evidence else None
        rows.append(
            {
                "source_evidence": evidence.text if evidence else "",
                "source_page": evidence.page if evidence else None,
                "extracted_item": interface.interface_id or interface.name,
                "item_type": "Interface",
                "sysml_element": interface.interface_id or interface.name,
                "rule_id": interface.related_rule_id or "O8",
                "confidence": 0.8 if evidence else 0.45,
            }
        )
    return rows


def _sysml_graph_metrics(
    analysis: ConOpsAnalysis,
    nodes_count: int,
    edges_count: int,
    grounded: int,
) -> dict:
    return {
        "nodes_count": nodes_count,
        "edges_count": edges_count,
        "requirements_count": len(analysis.requirements),
        "stakeholders_count": len(analysis.stakeholders),
        "risks_count": len(analysis.risks),
        "interfaces_count": len(analysis.interfaces),
        "grounded_nodes_count": grounded,
        "ungrounded_nodes_count": max(0, nodes_count - grounded),
        "evidence_coverage": round(grounded / nodes_count, 3)
        if nodes_count
        else 0.0,
        "validation_notice": "SysML v2 genere automatiquement - validation heuristique non officielle",
    }


def save_sysml_graph(
    analysis: ConOpsAnalysis,
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sysml_path = output / "sysml_model.sysml"
    graph_path = output / "sysml_graph.html"
    matrix_path = output / "sysml_traceability_matrix.csv"
    metrics_path = output / "sysml_graph_metrics.json"

    save_sysml(sysml_path, analysis)
    rows = sysml_traceability_rows(analysis)
    pd.DataFrame(rows).to_csv(matrix_path, index=False, encoding="utf-8-sig")

    network = Network(
        height="720px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#111111",
        directed=True,
    )
    node_ids: set[str] = set()
    edge_count = 0

    def add_node(node_id: str, label: str, group: str) -> None:
        if node_id not in node_ids:
            network.add_node(node_id, label=label, group=group)
            node_ids.add(node_id)

    def add_edge(source: str, target: str, label: str) -> None:
        nonlocal edge_count
        network.add_edge(source, target, label=label)
        edge_count += 1

    for row in rows:
        evidence_id = f"EVID-{len(network.nodes)}"
        item_id = str(row["extracted_item"])
        sysml_id = f"SYSML-{row['sysml_element']}"
        rule_id = f"RULE-{row['rule_id']}"
        add_node(evidence_id, "Evidence", "Evidence")
        add_node(item_id, item_id, row["item_type"])
        add_node(sysml_id, str(row["sysml_element"]), "SysML")
        add_node(rule_id, row["rule_id"], "Rule")
        add_edge(evidence_id, item_id, "supports")
        add_edge(item_id, sysml_id, "generated_as")
        add_edge(rule_id, sysml_id, "evaluates")
        if row["item_type"] == "Requirement":
            add_edge(item_id, evidence_id, "derived_from")
    network.add_node(
        "NOTICE",
        label="SysML v2 genere automatiquement - validation heuristique non officielle.",
        shape="box",
        physics=False,
    )
    node_ids.add("NOTICE")
    for index, need in enumerate(analysis.stakeholder_needs, start=1):
        need_id = f"NEED-{index:03d}"
        add_node(need_id, need[:60], "StakeholderNeed")
        for requirement in analysis.requirements[:3]:
            add_edge(need_id, requirement.id, "justifies")
    for requirement in analysis.requirements:
        system_id = f"SYS-{requirement.id}"
        add_node(system_id, "SystemElement", "SystemElement")
        add_edge(requirement.id, system_id, "satisfied_by")
    for interface in analysis.interfaces:
        interface_id = interface.interface_id or interface.name or "Interface"
        add_node(interface_id, interface_id, "Interface")
        for actor in (interface.source_actor, interface.target_actor):
            if actor:
                system_id = f"SYS-{actor}"
                add_node(system_id, actor, "SystemElement")
                add_edge(interface_id, system_id, "connects")
    for risk_index, risk in enumerate(analysis.risks, start=1):
        risk_id = f"RISK-{risk_index:03d}"
        add_node(risk_id, risk.description[:60], "Risk")
        for requirement in analysis.requirements[:3]:
            add_edge(risk_id, requirement.id, "impacts")
    network.write_html(str(graph_path), notebook=False)
    grounded = sum(1 for row in rows if row.get("source_evidence"))
    metrics_path.write_text(
        __import__("json").dumps(
            _sysml_graph_metrics(analysis, len(node_ids), edge_count, grounded),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "sysml_model": sysml_path,
        "sysml_graph": graph_path,
        "sysml_traceability_matrix": matrix_path,
        "sysml_graph_metrics": metrics_path,
    }


def generate_sysml_graph_exports(
    analysis: ConOpsAnalysis,
    output_dir: str | Path,
) -> dict[str, Path]:
    return save_sysml_graph(analysis, output_dir)
