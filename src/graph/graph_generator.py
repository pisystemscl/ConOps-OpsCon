from __future__ import annotations

import re
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from pyvis.network import Network

from src.core.schema import ConOpsAnalysis
from src.pipeline.quality import normalize_actor_name


TYPE_COLORS = {
    "Document": "#1f77b4",
    "ProblemSpace": "#9467bd",
    "FutureOperationalSoS": "#8c564b",
    "Stakeholder": "#17becf",
    "StakeholderNeed": "#bcbd22",
    "Requirement": "#ff7f0e",
    "Capability": "#2ca02c",
    "OperationalService": "#98df8a",
    "ConstituentSystem": "#7f7f7f",
    "Interface": "#e377c2",
    "OperationalScenario": "#c5b0d5",
    "OperationalFlow": "#9edae5",
    "Risk": "#d62728",
    "FutureAction": "#aec7e8",
    "ExpectedOutcome": "#dbdb8d",
    "Enabler": "#c49c94",
}


def _tokens(text: str) -> set[str]:
    stop_words = {
        "with", "from", "that", "this", "shall", "system", "service",
        "pour", "avec", "dans", "doit", "the", "and", "des", "les",
    }
    return {
        token.lower()
        for token in re.findall(r"[A-Za-zÀ-ÿ]{4,}", text)
        if token.lower() not in stop_words
    }


def _relation_confidence(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(
        len(left_tokens),
        len(right_tokens),
    )


def _first_evidence(item) -> tuple[str, int | None]:
    evidence = item.evidence[0] if getattr(item, "evidence", None) else None
    return (
        (evidence.text if evidence else ""),
        (evidence.page if evidence else None),
    )


def _add_node(
    graph: nx.DiGraph,
    node_id: str,
    node_type: str,
    label: str,
    title: str,
    *,
    evidence_quote: str = "",
    source_page: int | None = None,
    source_document: str = "",
    related_rule_id: str = "",
    confidence: float = 1.0,
    validation_status: str | None = None,
) -> None:
    grounded = bool(evidence_quote and source_page is not None)
    graph.add_node(
        node_id,
        group=node_type,
        node_type=node_type,
        label=label,
        title=title,
        confidence=confidence if grounded else min(confidence, 0.55),
        validation_status=validation_status
        or ("automatic_grounded" if grounded else "human_review_required"),
        source_page=source_page,
        evidence_quote=evidence_quote,
        source_document=source_document,
        related_rule_id=related_rule_id,
        color=TYPE_COLORS.get(node_type, "#aaaaaa"),
    )


def _add_edge(
    graph: nx.DiGraph,
    source: str,
    target: str,
    relation: str,
    *,
    evidence_quote: str = "",
    source_page: int | None = None,
) -> None:
    grounded = bool(evidence_quote and source_page is not None)
    graph.add_edge(
        source,
        target,
        label=relation,
        relation=relation,
        relation_type=relation,
        source_id=source,
        target_id=target,
        confidence=1.0 if grounded else 0.6,
        evidence_quote=evidence_quote,
        source_page=source_page,
        title=(
            f"{relation}<br>{evidence_quote[:250]}"
            if evidence_quote
            else relation
        ),
    )


def _source_document(item) -> str:
    evidence = item.evidence[0] if getattr(item, "evidence", None) else None
    return evidence.source if evidence else ""


def build_graph(analysis: ConOpsAnalysis) -> nx.DiGraph:
    graph = nx.DiGraph()
    classification = analysis.document_classification or "Neither"
    _add_node(
        graph,
        "DOCUMENT",
        "Document",
        analysis.document_title or "Document",
        f"{classification}<br>{analysis.summary or analysis.document_title}",
        validation_status="automatic_context",
    )
    if classification in {"ConOps", "Hybrid"}:
        _add_node(
            graph,
            "PROBLEM-SPACE",
            "ProblemSpace",
            "Problem Space",
            "Official V3.0 ConOps orientation",
        )
        _add_edge(graph, "DOCUMENT", "PROBLEM-SPACE", "describes")
    if classification in {"OpsCon", "Hybrid"}:
        _add_node(
            graph,
            "FUTURE-SOS",
            "FutureOperationalSoS",
            "Future Operational SoS",
            "Official V3.0 OpsCon concept",
        )
        _add_edge(graph, "DOCUMENT", "FUTURE-SOS", "describes")

    stakeholders: dict[str, str] = {}
    stakeholder_items = []
    for index, stakeholder in enumerate(analysis.stakeholders, start=1):
        node_id = f"STK-{index:03d}"
        actor_name = normalize_actor_name(stakeholder.name)
        evidence, page = _first_evidence(stakeholder)
        stakeholders[actor_name.lower()] = node_id
        stakeholder_items.append((node_id, stakeholder))
        _add_node(
            graph,
            node_id,
            "Stakeholder",
            actor_name,
            f"{stakeholder.role}<br>{stakeholder.interest}",
            evidence_quote=evidence,
            source_page=page,
            source_document=_source_document(stakeholder),
            related_rule_id="C2",
        )
        _add_edge(
            graph,
            "DOCUMENT",
            node_id,
            "involves",
            evidence_quote=evidence,
            source_page=page,
        )

    needs = []
    for index, need in enumerate(analysis.stakeholder_needs, start=1):
        node_id = f"NEED-{index:03d}"
        needs.append((node_id, need))
        _add_node(graph, node_id, "StakeholderNeed", node_id, need)
        _add_edge(graph, "DOCUMENT", node_id, "captures_need")
        for stakeholder_id, stakeholder in stakeholder_items:
            if stakeholder.name.lower() in need.lower():
                _add_edge(
                    graph,
                    stakeholder_id,
                    node_id,
                    "expresses",
                    evidence_quote=need,
                )

    for index, outcome in enumerate(analysis.expected_outcomes, start=1):
        node_id = f"OUT-{index:03d}"
        _add_node(graph, node_id, "ExpectedOutcome", node_id, outcome)
        _add_edge(graph, "DOCUMENT", node_id, "targets_outcome")

    services = []
    for index, service in enumerate(analysis.expected_services, start=1):
        node_id = f"SRV-{index:03d}"
        services.append((node_id, service))
        _add_node(
            graph,
            node_id,
            "OperationalService",
            node_id,
            service,
        )
        _add_edge(graph, "DOCUMENT", node_id, "defines_service")

    capabilities = []
    for index, capability in enumerate(analysis.capabilities, start=1):
        node_id = f"CAP-{index:03d}"
        capabilities.append((node_id, capability))
        _add_node(graph, node_id, "Capability", node_id, capability)
        _add_edge(graph, "DOCUMENT", node_id, "requires_capability")

    for index, enabler in enumerate(analysis.enablers, start=1):
        node_id = f"ENB-{index:03d}"
        _add_node(graph, node_id, "Enabler", node_id, enabler)
        _add_edge(graph, "DOCUMENT", node_id, "uses_enabler")

    requirements = []
    for index, requirement in enumerate(analysis.requirements, start=1):
        node_id = requirement.id or f"REQ-{index:03d}"
        evidence, page = _first_evidence(requirement)
        requirements.append((node_id, requirement))
        _add_node(
            graph,
            node_id,
            "Requirement",
            node_id,
            requirement.text,
            evidence_quote=evidence,
            source_page=page,
            source_document=_source_document(requirement),
            related_rule_id="C5",
        )
        _add_edge(
            graph,
            "DOCUMENT",
            node_id,
            "specified_by",
            evidence_quote=evidence,
            source_page=page,
        )
        for need_id, need in needs:
            if _relation_confidence(need, requirement.text) >= 0.45:
                _add_edge(
                    graph,
                    need_id,
                    node_id,
                    "justifies",
                    evidence_quote=evidence,
                    source_page=page,
                )
        for capability_id, capability in capabilities:
            if _relation_confidence(requirement.text, capability) >= 0.45:
                _add_edge(
                    graph,
                    node_id,
                    capability_id,
                    "realizes",
                    evidence_quote=evidence,
                    source_page=page,
                )

    risks = []
    for index, risk in enumerate(analysis.risks, start=1):
        node_id = f"RISK-{index:03d}"
        evidence, page = _first_evidence(risk)
        risks.append((node_id, risk))
        _add_node(
            graph,
            node_id,
            "Risk",
            node_id,
            f"{risk.description}<br>Mitigation: {risk.mitigation}",
            evidence_quote=evidence,
            source_page=page,
            source_document=_source_document(risk),
            related_rule_id="C13",
        )
        _add_edge(
            graph,
            "DOCUMENT",
            node_id,
            "exposed_to",
            evidence_quote=evidence,
            source_page=page,
        )
        for requirement_id, requirement in requirements:
            if _relation_confidence(
                risk.description,
                requirement.text,
            ) >= 0.45:
                _add_edge(
                    graph,
                    requirement_id,
                    node_id,
                    "constrained_by",
                    evidence_quote=evidence,
                    source_page=page,
                )
                _add_edge(
                    graph,
                    node_id,
                    requirement_id,
                    "threatens",
                    evidence_quote=evidence,
                    source_page=page,
                )

    actions = []
    for index, action in enumerate(analysis.future_actions, start=1):
        node_id = f"ACT-{index:03d}"
        actions.append((node_id, action))
        _add_node(
            graph,
            node_id,
            "FutureAction",
            node_id,
            f"{action.action}<br>Owner: {action.owner}",
        )
        _add_edge(graph, "DOCUMENT", node_id, "plans")
        for requirement_id, requirement in requirements:
            if _relation_confidence(
                action.action,
                requirement.text,
            ) >= 0.45:
                _add_edge(
                    graph,
                    node_id,
                    requirement_id,
                    "improves",
                    evidence_quote=action.rationale,
                )
        for risk_id, risk in risks:
            if risk.mitigation and _relation_confidence(
                risk.mitigation,
                action.action,
            ) >= 0.45:
                _add_edge(
                    graph,
                    risk_id,
                    node_id,
                    "mitigated_by",
                    evidence_quote=risk.mitigation,
                )

    for index, interface in enumerate(analysis.interfaces, start=1):
        node_id = f"INT-{index:03d}"
        evidence, page = _first_evidence(interface)
        _add_node(
            graph,
            node_id,
            "Interface",
            node_id,
            interface.exchanged_information,
            evidence_quote=evidence,
            source_page=page,
            source_document=_source_document(interface),
            related_rule_id=interface.related_rule_id or "O8",
        )
        for actor_name, relation in (
            (interface.source, "connects_source"),
            (interface.target, "connects_target"),
        ):
            normalized = normalize_actor_name(actor_name)
            actor_id = stakeholders.get(normalized.lower())
            if actor_id is None and normalized:
                actor_id = f"ACTOR-{len(stakeholders) + 1:03d}"
                stakeholders[normalized.lower()] = actor_id
                _add_node(
                    graph,
                    actor_id,
                    "Stakeholder",
                    normalized,
                    normalized,
                )
            if actor_id:
                _add_edge(
                    graph,
                    node_id,
                    actor_id,
                    relation,
                    evidence_quote=evidence,
                    source_page=page,
                )

    constituents = []
    if classification in {"OpsCon", "Hybrid"}:
        for index, system in enumerate(
            analysis.constituent_systems,
            start=1,
        ):
            node_id = f"CS-{index:03d}"
            constituents.append((node_id, system))
            _add_node(
                graph,
                node_id,
                "ConstituentSystem",
                node_id,
                system,
            )
            _add_edge(
                graph,
                "FUTURE-SOS",
                node_id,
                "has_constituent_system",
            )
        for service_id, service in services:
            for system_id, system in constituents:
                if system.lower() in service.lower():
                    _add_edge(
                        graph,
                        service_id,
                        system_id,
                        "supported_by",
                        evidence_quote=service,
                    )

        flows = []
        for index, flow in enumerate(analysis.operational_flows, start=1):
            node_id = f"FLOW-{index:03d}"
            flows.append((node_id, flow))
            _add_node(
                graph,
                node_id,
                "OperationalFlow",
                node_id,
                flow,
            )
            _add_edge(graph, "FUTURE-SOS", node_id, "has_flow")

        scenarios = [
            item
            for item in analysis.opscon_elements
            if any(
                term in item.lower()
                for term in ("scenario", "workflow", "use case", "thread")
            )
        ]
        for index, scenario in enumerate(scenarios, start=1):
            scenario_id = f"SCN-{index:03d}"
            _add_node(
                graph,
                scenario_id,
                "OperationalScenario",
                scenario_id,
                scenario,
            )
            for flow_id, flow in flows:
                if _relation_confidence(scenario, flow) >= 0.45:
                    _add_edge(
                        graph,
                        scenario_id,
                        flow_id,
                        "contains",
                        evidence_quote=scenario,
                    )

    for capability_id, capability in capabilities:
        for service_id, service in services:
            if _relation_confidence(capability, service) >= 0.45:
                _add_edge(
                    graph,
                    capability_id,
                    service_id,
                    "enables",
                    evidence_quote=capability,
                )
    return graph


def graph_metrics(graph: nx.DiGraph) -> dict:
    nodes_count = graph.number_of_nodes()
    edges_count = graph.number_of_edges()
    isolated = list(nx.isolates(graph.to_undirected()))
    evidence_nodes = sum(
        bool(data.get("evidence_quote") and data.get("source_page") is not None)
        for _, data in graph.nodes(data=True)
    )
    confidence_values = [
        float(data.get("confidence", 0.0))
        for _, data in graph.nodes(data=True)
    ] + [
        float(data.get("confidence", 0.0))
        for _, _, data in graph.edges(data=True)
    ]
    missing_pages = sum(
        data.get("source_page") is None
        for _, data in graph.nodes(data=True)
    )
    undirected = graph.to_undirected()
    return {
        "nodes_count": nodes_count,
        "edges_count": edges_count,
        "isolated_nodes_count": len(isolated),
        "evidence_coverage": round(evidence_nodes / nodes_count, 3)
        if nodes_count
        else 0.0,
        "grounded_nodes_count": evidence_nodes,
        "ungrounded_nodes_count": max(0, nodes_count - evidence_nodes),
        "average_confidence": round(
            sum(confidence_values) / len(confidence_values),
            3,
        )
        if confidence_values
        else 0.0,
        "missing_source_pages_count": missing_pages,
        "graph_density": round(nx.density(graph), 3) if nodes_count > 1 else 0.0,
        "connected_components_count": nx.number_connected_components(undirected)
        if nodes_count
        else 0,
    }


def graph_data(graph: nx.DiGraph) -> dict:
    return {
        "nodes": [
            {"id": node_id, **attributes}
            for node_id, attributes in graph.nodes(data=True)
        ],
        "edges": [
            {"source": source, "target": target, **attributes}
            for source, target, attributes in graph.edges(data=True)
        ],
    }


def save_graph_data(
    analysis: ConOpsAnalysis,
    data_path: str | Path,
    metrics_path: str | Path,
) -> tuple[Path, Path]:
    import json

    graph = build_graph(analysis)
    data_output = Path(data_path)
    metrics_output = Path(metrics_path)
    data_output.parent.mkdir(parents=True, exist_ok=True)
    data_output.write_text(
        json.dumps(graph_data(graph), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics_output.write_text(
        json.dumps(graph_metrics(graph), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return data_output, metrics_output


def graph_edges_dataframe(
    analysis: ConOpsAnalysis,
    *,
    run_id: str = "",
    configuration: str = "",
) -> pd.DataFrame:
    graph = build_graph(analysis)
    rows = []
    for source, target, attributes in graph.edges(data=True):
        rows.append(
            {
                "run_id": run_id,
                "configuration": configuration,
                "source": source,
                "target": target,
                "relation": attributes.get("relation", ""),
                "source_type": graph.nodes[source].get("node_type", ""),
                "target_type": graph.nodes[target].get("node_type", ""),
                "evidence_quote": attributes.get("evidence_quote", ""),
                "source_page": attributes.get("source_page"),
            }
        )
    return pd.DataFrame(rows)


def save_graph_edges_csv(
    analysis: ConOpsAnalysis,
    output_path: str | Path,
    *,
    run_id: str = "",
    configuration: str = "",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph_edges_dataframe(
        analysis,
        run_id=run_id,
        configuration=configuration,
    ).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_pyvis_graph(
    analysis: ConOpsAnalysis,
    output_path: str | Path,
) -> Path:
    graph = build_graph(analysis)
    network = Network(
        height="750px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#111111",
        directed=True,
    )
    network.from_nx(graph)
    for index, (node_type, color) in enumerate(TYPE_COLORS.items()):
        network.add_node(
            f"LEGEND-{node_type}",
            label=node_type,
            color=color,
            shape="box",
            physics=False,
            fixed=True,
            x=-950,
            y=-420 + index * 55,
        )
    network.repulsion(
        node_distance=190,
        central_gravity=0.2,
        spring_length=190,
        spring_strength=0.05,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    network.write_html(str(output_path), notebook=False)
    return output_path


def save_3d_graph(
    analysis: ConOpsAnalysis,
    output_path: str | Path,
) -> Path:
    graph = build_graph(analysis)
    positions = nx.spring_layout(graph, dim=3, seed=42)
    figure = go.Figure()
    edge_x, edge_y, edge_z = [], [], []
    for source, target in graph.edges():
        x0, y0, z0 = positions[source]
        x1, y1, z1 = positions[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]
    figure.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(width=2, color="#888888"),
            hoverinfo="none",
            showlegend=False,
        )
    )
    groups = sorted(
        {attributes.get("node_type", "Other") for _, attributes in graph.nodes(data=True)}
    )
    for group in groups:
        nodes = [
            node
            for node, attributes in graph.nodes(data=True)
            if attributes.get("node_type", "Other") == group
        ]
        figure.add_trace(
            go.Scatter3d(
                x=[positions[node][0] for node in nodes],
                y=[positions[node][1] for node in nodes],
                z=[positions[node][2] for node in nodes],
                mode="markers+text",
                name=group,
                text=[graph.nodes[node].get("label", node) for node in nodes],
                customdata=[
                    graph.nodes[node].get("title", node) for node in nodes
                ],
                hovertemplate="%{customdata}<extra></extra>",
                textposition="top center",
                marker=dict(
                    size=6,
                    color=TYPE_COLORS.get(group, "#aaaaaa"),
                ),
            )
        )
    figure.update_layout(
        title=f"Graphe de connaissances - {analysis.document_title}",
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_path))
    return output_path
