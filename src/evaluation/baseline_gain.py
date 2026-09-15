from __future__ import annotations

import pandas as pd


GAIN_COLUMNS = [
    "configuration",
    "final_score",
    "gain_vs_C1",
    "conops_score",
    "opscon_score",
    "sysml_score",
    "semantic_similarity",
    "grounding_score",
    "rag_used",
    "llm_evaluator_used",
    "human_review_rate",
]


def gain_vs_baseline(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty or "configuration" not in dataframe:
        return pd.DataFrame(columns=GAIN_COLUMNS)
    successful = (
        dataframe[dataframe["status"] == "success"].copy()
        if "status" in dataframe
        else dataframe.copy()
    )
    baseline_rows = successful[successful["configuration"] == "C1"]
    baseline = (
        float(baseline_rows.iloc[0]["final_score"])
        if not baseline_rows.empty
        else None
    )
    successful["gain_vs_C1"] = (
        successful["final_score"] - baseline
        if baseline is not None
        else None
    )
    for column in GAIN_COLUMNS:
        if column not in successful:
            successful[column] = None
    return successful[GAIN_COLUMNS]


def interpret_baseline_gain(dataframe: pd.DataFrame) -> list[str]:
    gain = gain_vs_baseline(dataframe)
    if gain.empty:
        return ["Comparaison impossible: aucun résultat exploitable."]
    by_code = gain.set_index("configuration")
    messages = []
    if {"C1", "C4"}.issubset(by_code.index):
        improved = (
            by_code.loc["C4", "grounding_score"]
            > by_code.loc["C1", "grounding_score"]
        )
        messages.append(
            "C4 améliore le grounding par rapport à C1."
            if improved
            else "C4 n'améliore pas le grounding sur ce document."
        )
    if {"C1", "C5"}.issubset(by_code.index):
        delta = float(by_code.loc["C5", "gain_vs_C1"])
        messages.append(
            "Gain faible : l'apport du RAG + évaluateur reste à confirmer "
            "sur plusieurs documents."
            if delta < 2
            else f"C5 améliore le score final de {delta:.2f} points."
        )
    if {"C1", "C3"}.issubset(by_code.index):
        delta = float(by_code.loc["C3", "gain_vs_C1"])
        messages.append(
            "C3 est pénalisé par l'application stricte des règles."
            if delta < 0
            else "C3 bénéficie des règles sans pénalité nette."
        )
    return messages
