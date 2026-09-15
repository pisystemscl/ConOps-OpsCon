from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any

from src.llm.json_repair import safe_extract_json


def slugify(value: str, max_len: int = 60) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9A-Zàâäéèêëïîôöùûüç_ -]+", "", value)
    value = re.sub(r"\s+", "_", value)
    return value[:max_len] or "document"


def save_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

