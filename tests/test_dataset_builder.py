import json

import pandas as pd

from src.training.dataset_builder import build_dataset


def test_dataset_builder_creates_train_validation_split(tmp_path):
    expert_validated_folder = tmp_path / "expert_validated"
    expert_validated_folder.mkdir()
    for index in range(3):
        (expert_validated_folder / f"validated_{index}.json").write_text(
            json.dumps(
                {
                    "source_text": f"Validated source {index}",
                    "analysis": {"document_title": f"Document {index}"},
                }
            ),
            encoding="utf-8",
        )
    expert_path = tmp_path / "expert_validation.csv"
    pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "configuration": "C5",
                "item_id": "REQ-001",
                "item_type": "requirement",
                "item_text": "Share operational data",
                "evidence_quote": "Operational data shall be shared.",
                "source_page": 4,
                "expert_label": "correct",
                "expert_comment": "",
            }
        ]
    ).to_csv(expert_path, index=False)

    result = build_dataset(
        expert_validated_folder=expert_validated_folder,
        expert_validation_paths=[expert_path],
        output_dir=tmp_path / "dataset",
        validation_ratio=0.25,
    )

    assert result["examples"] == 4
    assert result["train"].exists()
    assert result["validation"].exists()
    assert result["validation"].read_text(encoding="utf-8").strip()
