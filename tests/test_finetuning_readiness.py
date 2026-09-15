import json

import pytest

from src.llm.fine_tuned_provider import fine_tuning_readiness
from src.training.train_lora import validate_jsonl_dataset


def test_fine_tuning_readiness_reports_missing_adapter_and_dataset(tmp_path):
    readiness = fine_tuning_readiness(
        adapter_path=tmp_path / "missing_adapter",
        dataset_path=tmp_path / "missing_dataset.jsonl",
        minimum_examples=2,
    )

    assert readiness.is_ready is False
    assert readiness.dataset_examples_count == 0
    assert "Adaptateur LoRA introuvable" in readiness.message()
    assert "Dataset expert insuffisant" in readiness.message()


def test_validate_jsonl_dataset_requires_enough_valid_examples(tmp_path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "instruction": "Extract",
                "input": "Document",
                "output": {"items": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Dataset insuffisant"):
        validate_jsonl_dataset(dataset, min_examples=2)

    assert validate_jsonl_dataset(dataset, min_examples=1) == 1


def test_validate_jsonl_dataset_rejects_missing_required_fields(tmp_path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        json.dumps({"instruction": "Extract", "input": "Document"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="champs manquants"):
        validate_jsonl_dataset(dataset, min_examples=1)
