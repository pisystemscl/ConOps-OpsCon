from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

EXPERT_COLUMNS = [
    "run_id", "configuration", "item_id", "item_type", "item_text",
    "evidence_quote", "source_page", "auto_score", "expert_label",
    "expert_comment", "timestamp",
]


def build_expert_validation_frame(
    detailed: pd.DataFrame,
    configuration: str | None = None,
) -> pd.DataFrame:
    frame = detailed.copy()
    if configuration:
        frame = frame[frame["configuration"] == configuration].copy()
    result = pd.DataFrame(
        {
            "run_id": frame.get("run_id", ""),
            "configuration": frame.get("configuration", ""),
            "item_id": frame.get("item_id", ""),
            "item_type": frame.get("item_type", ""),
            "item_text": frame.get("item_text", ""),
            "evidence_quote": frame.get("evidence_quote", ""),
            "source_page": frame.get("source_page"),
            "auto_score": frame.get("item_quality_score", 0.0),
            "expert_label": "a_revoir",
            "expert_comment": "",
            "timestamp": "",
        }
    )
    return result[EXPERT_COLUMNS]


def save_expert_validation(
    frame: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output["timestamp"] = datetime.now(timezone.utc).isoformat()
    output = output.reindex(columns=EXPERT_COLUMNS)
    output.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def expert_agreement_score(frame: pd.DataFrame) -> float | None:
    if frame.empty or "expert_label" not in frame:
        return None
    reviewed = frame[frame["expert_label"].isin(["correct", "partiellement_correct", "faux"])]
    if reviewed.empty:
        return None
    score_map = {"correct": 1.0, "partiellement_correct": 0.5, "faux": 0.0}
    return round(reviewed["expert_label"].map(score_map).mean(), 3)
