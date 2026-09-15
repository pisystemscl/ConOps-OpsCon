from __future__ import annotations

import pandas as pd


def explain_best_configuration(frame: pd.DataFrame, best_code: str) -> dict:
    successful = frame[frame["status"] == "success"].copy()
    if successful.empty or best_code not in set(successful["configuration"]):
        return {"bullets": ["Aucune configuration gagnante exploitable."], "gain_vs_C1": None}
    best = successful[successful["configuration"] == best_code].iloc[0]
    c1_rows = successful[successful["configuration"] == "C1"]
    c1 = c1_rows.iloc[0] if not c1_rows.empty else None
    gain = (
        float(best.get("final_score", 0.0) or 0.0)
        - float(c1.get("final_score", 0.0) or 0.0)
        if c1 is not None
        else None
    )
    grounding_gain = (
        float(best.get("grounding_score", 0.0) or 0.0)
        - float(c1.get("grounding_score", 0.0) or 0.0)
        if c1 is not None
        else None
    )
    semantic_gain = (
        float(best.get("semantic_similarity", 0.0) or 0.0)
        - float(c1.get("semantic_similarity", 0.0) or 0.0)
        if c1 is not None
        else None
    )
    bullets = [
        f"Gain vs C1: {gain:.2f} point(s)." if gain is not None else "C1 indisponible pour calculer le gain.",
        f"Grounding improvement: {grounding_gain:.3f}." if grounding_gain is not None else "Grounding improvement non calculable.",
        f"Semantic similarity improvement: {semantic_gain:.3f}." if semantic_gain is not None else "Semantic similarity improvement non calculable.",
        "RAG active." if bool(best.get("rag_used")) else "RAG non actif.",
        "Evaluateur complementaire actif." if bool(best.get("llm_evaluator_used")) else "Evaluateur complementaire non actif.",
        "Conclusion limitee: a confirmer avec plusieurs repetitions et plusieurs documents.",
    ]
    if gain is not None and gain < 3:
        bullets.append(
            "Gain faible : l'apport de cette configuration reste a confirmer sur plusieurs documents."
        )
    return {
        "best_configuration": best_code,
        "gain_vs_C1": gain,
        "grounding_improvement": grounding_gain,
        "semantic_similarity_improvement": semantic_gain,
        "bullets": bullets,
    }
