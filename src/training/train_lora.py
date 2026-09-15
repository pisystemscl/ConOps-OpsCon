from __future__ import annotations
import argparse
import json
from pathlib import Path


REQUIRED_JSONL_FIELDS = {"instruction", "input", "output"}


def validate_jsonl_dataset(path: str | Path, *, min_examples: int = 20) -> int:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {dataset_path}")
    if dataset_path.stat().st_size == 0:
        raise ValueError(f"Dataset vide: {dataset_path}")
    count = 0
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL invalide ligne {line_number}: {exc}"
                ) from exc
            missing = REQUIRED_JSONL_FIELDS - set(row)
            if missing:
                raise ValueError(
                    f"Dataset invalide ligne {line_number}: champs manquants "
                    f"{sorted(missing)}"
                )
            count += 1
    if count < min_examples:
        raise ValueError(
            f"Dataset insuffisant pour un fine-tuning credible: "
            f"{count}/{min_examples} exemples valides."
        )
    return count


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA pour extraction ConOps -> JSON")
    parser.add_argument("--model_name", required=True, help="Exemple : TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--dataset", default="data/fine_tuning/train_dataset.jsonl")
    parser.add_argument(
        "--validation_dataset",
        default="data/fine_tuning/validation_dataset.jsonl",
    )
    parser.add_argument("--output_dir", default="models/conops-lora")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_examples", type=int, default=20)
    args = parser.parse_args()

    train_count = validate_jsonl_dataset(
        args.dataset,
        min_examples=args.min_examples,
    )
    if (
        Path(args.validation_dataset).exists()
        and Path(args.validation_dataset).stat().st_size > 0
    ):
        validate_jsonl_dataset(
            args.validation_dataset,
            min_examples=1,
        )
    print(f"Dataset expert valide: {train_count} exemples d'entrainement.")

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig
    from trl import SFTTrainer

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    validation_dataset = (
        load_dataset(
            "json",
            data_files=args.validation_dataset,
            split="train",
        )
        if Path(args.validation_dataset).exists()
        and Path(args.validation_dataset).stat().st_size > 0
        else None
    )

    def format_example(example):
        return (
            "### Instruction:\n" + example["instruction"] +
            "\n\n### Input:\n" + example["input"] +
            "\n\n### Output JSON:\n" + str(example["output"])
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="epoch",
        fp16=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=validation_dataset,
        formatting_func=format_example,
        peft_config=peft_config,
        args=training_args,
        max_seq_length=2048,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Modèle LoRA sauvegardé dans {args.output_dir}")

if __name__ == "__main__":
    main()
