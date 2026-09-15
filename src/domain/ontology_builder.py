from __future__ import annotations

CONOPS_CONCEPTS = [
    "ProblemSpace",
    "CurrentSituation",
    "Stakeholder",
    "StakeholderNeed",
    "ExpectedOutcome",
    "OperationalEnvironment",
    "ChangeDriver",
    "FutureService",
    "CapabilityGap",
    "FutureCapability",
    "FutureSolutionConstraint",
    "SharedOperationalVision",
    "OpportunityArea",
]

OPSCON_CONCEPTS = [
    "SolutionSpace",
    "FutureOperationalSoS",
    "SoSBoundary",
    "ConstituentSystem",
    "PromoterSystemOfInterest",
    "OperationalService",
    "OperationalScenario",
    "OperationalFlow",
    "Interface",
    "MEIFlow",
    "Governance",
    "OperationalDependency",
    "OperationalOwnership",
    "OperationalMode",
    "Promoter",
    "Alliance",
]

CONOPS_RELATIONS = [
    ("Stakeholder", "expresses", "StakeholderNeed"),
    ("StakeholderNeed", "leads_to", "ExpectedOutcome"),
    ("ChangeDriver", "justifies", "FutureService"),
    ("CapabilityGap", "motivates", "FutureCapability"),
    ("FutureSolutionConstraint", "constrains", "FutureOpsCon"),
    ("FutureService", "remains_black_box_in", "ConOps"),
]

OPSCON_RELATIONS = [
    ("FutureOperationalSoS", "has", "ConstituentSystem"),
    ("SoSBoundary", "defines", "inside_outside"),
    ("ConstituentSystem", "realizes", "OperationalService"),
    ("ConstituentSystem", "exchanges", "MEIFlow"),
    ("MEIFlow", "passes_through", "Interface"),
    ("Governance", "coordinates", "ConstituentSystem"),
    ("OperationalDependency", "links", "ConstituentSystem"),
    ("Promoter", "forms", "Alliance"),
    ("OperationalService", "traces_to", "StakeholderNeed"),
]


def build_ontology() -> dict:
    return {
        "conops": {
            "concepts": list(CONOPS_CONCEPTS),
            "relations": list(CONOPS_RELATIONS),
        },
        "opscon": {
            "concepts": list(OPSCON_CONCEPTS),
            "relations": list(OPSCON_RELATIONS),
        },
    }
