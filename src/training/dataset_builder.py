from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


INSTRUCTION = (
    "Extract ConOps/OpsCon elements and return valid structured JSON with "
    "short evidence quotes. Facts must be grounded in the source text."
)


def _read_json_examples(folder: Path) -> list[dict]:
    examples = []
    if not folder.exists():
        return examples
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        source_text = str(
            data.get("source_text")
            or data.get("document_text")
            or data.get("input")
            or ""
        ).strip()
        output = data.get("analysis") or data.get("output")
        if source_text and isinstance(output, dict):
            examples.append(
                {
                    "instruction": INSTRUCTION,
                    "input": source_text,
                    "output": output,
                    "source": str(path),
                }
            )
    return examples


def _read_expert_examples(paths: list[Path]) -> list[dict]:
    examples = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "expert_label" not in frame.columns:
            continue
        accepted = frame[
            frame["expert_label"].isin(
                ["correct", "partiellement_correct"]
            )
        ].copy()
        if accepted.empty:
            continue
        for (_, configuration), group in accepted.groupby(
            ["run_id", "configuration"],
            dropna=False,
        ):
            items = [
                {
                    "item_id": row.get("item_id", ""),
                    "item_type": row.get("item_type", ""),
                    "item_text": row.get("item_text", ""),
                    "evidence_quote": row.get("evidence_quote", ""),
                    "source_page": row.get("source_page"),
                    "expert_label": row.get("expert_label", ""),
                    "expert_comment": row.get("expert_comment", ""),
                }
                for _, row in group.iterrows()
            ]
            source_text = "\n".join(
                str(item["evidence_quote"])
                for item in items
                if item["evidence_quote"]
            )
            if source_text:
                examples.append(
                    {
                        "instruction": INSTRUCTION,
                        "input": source_text,
                        "output": {"validated_items": items},
                        "source": str(path),
                    }
                )
    return examples


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def build_dataset(
    *,
    expert_validated_folder: str | Path = "data/expert_validated",
    expert_validation_paths: list[str | Path] | None = None,
    output_dir: str | Path = "data/fine_tuning",
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Path | int]:
    output_dir = Path(output_dir)
    expert_paths = [
        Path(path) for path in (expert_validation_paths or [])
    ]
    examples = _read_json_examples(Path(expert_validated_folder))
    examples.extend(_read_expert_examples(expert_paths))
    deduplicated = {
        json.dumps(item, sort_keys=True, ensure_ascii=False): item
        for item in examples
    }
    rows = list(deduplicated.values())
    random.Random(seed).shuffle(rows)
    if len(rows) >= 2:
        validation_count = max(1, round(len(rows) * validation_ratio))
    else:
        validation_count = 0
    validation_rows = rows[:validation_count]
    train_rows = rows[validation_count:]

    full_path = output_dir / "expert_validated_training_data.jsonl"
    train_path = output_dir / "train_dataset.jsonl"
    validation_path = output_dir / "validation_dataset.jsonl"
    _write_jsonl(full_path, rows)
    _write_jsonl(train_path, train_rows)
    _write_jsonl(validation_path, validation_rows)
    return {
        "full": full_path,
        "train": train_path,
        "validation": validation_path,
        "examples": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build validated train/validation JSONL datasets."
    )
    parser.add_argument(
        "--expert_validated_folder",
        default="data/expert_validated",
    )
    parser.add_argument(
        "--expert_validation",
        action="append",
        default=[],
    )
    parser.add_argument("--output_dir", default="data/fine_tuning")
    parser.add_argument("--validation_ratio", type=float, default=0.2)
    args = parser.parse_args()
    result = build_dataset(
        expert_validated_folder=args.expert_validated_folder,
        expert_validation_paths=args.expert_validation,
        output_dir=args.output_dir,
        validation_ratio=args.validation_ratio,
    )
    print(
        f"Dataset prepare: {result['examples']} examples; "
        f"train={result['train']}; validation={result['validation']}"
    )


if __name__ == "__main__":
    main()
