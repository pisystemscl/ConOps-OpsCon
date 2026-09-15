from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from src.config import PROJECT_DIR, settings
from src.llm.providers import BaseLLM


FINE_TUNING_NOT_AVAILABLE_MESSAGE = (
    "C6/C7 prepares mais non executes : aucun adaptateur fine-tuned valide disponible."
)


def fine_tuned_adapter_path() -> Path:
    path = Path(settings.finetuned_model_path)
    return path if path.is_absolute() else PROJECT_DIR / path


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


@dataclass(frozen=True)
class FineTuningReadiness:
    adapter_path: Path
    dataset_path: Path
    is_ready: bool
    reasons: tuple[str, ...]
    dataset_examples_count: int

    def message(self) -> str:
        if self.is_ready:
            return (
                "Adaptateur LoRA valide et dataset expert disponible; "
                "C6/C7 peuvent etre executes sous protocole controle."
            )
        return " ".join(self.reasons)


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def fine_tuning_readiness(
    adapter_path: str | Path | None = None,
    dataset_path: str | Path = "data/fine_tuning/expert_validated_training_data.jsonl",
    minimum_examples: int = 20,
) -> FineTuningReadiness:
    resolved_adapter = (
        _resolve_path(adapter_path) if adapter_path else fine_tuned_adapter_path()
    )
    resolved_dataset = _resolve_path(dataset_path)
    reasons: list[str] = []
    if not resolved_adapter.is_dir():
        reasons.append(f"Adaptateur LoRA introuvable: {resolved_adapter}.")
    if not (resolved_adapter / "adapter_config.json").exists():
        reasons.append("Fichier adapter_config.json absent.")
    if not any(resolved_adapter.glob("adapter_model.*")):
        reasons.append("Poids adapter_model.* absents.")
    examples_count = _count_jsonl_rows(resolved_dataset)
    if examples_count < minimum_examples:
        reasons.append(
            "Dataset expert insuffisant: "
            f"{examples_count}/{minimum_examples} exemples JSONL valides."
        )
    return FineTuningReadiness(
        adapter_path=resolved_adapter,
        dataset_path=resolved_dataset,
        is_ready=not reasons,
        reasons=tuple(reasons),
        dataset_examples_count=examples_count,
    )


def has_ready_fine_tuning(
    adapter_path: str | Path | None = None,
    dataset_path: str | Path = "data/fine_tuning/expert_validated_training_data.jsonl",
    minimum_examples: int = 20,
) -> bool:
    return fine_tuning_readiness(
        adapter_path=adapter_path,
        dataset_path=dataset_path,
        minimum_examples=minimum_examples,
    ).is_ready


def has_valid_fine_tuned_adapter(path: str | Path | None = None) -> bool:
    adapter_path = _resolve_path(path) if path else fine_tuned_adapter_path()
    return (
        adapter_path.is_dir()
        and (adapter_path / "adapter_config.json").exists()
        and any(adapter_path.glob("adapter_model.*"))
    )


class FineTunedLLM(BaseLLM):
    def __init__(self, adapter_path: str | Path | None = None):
        self.adapter_path = _resolve_path(adapter_path) if adapter_path else fine_tuned_adapter_path()
        if not has_valid_fine_tuned_adapter(self.adapter_path):
            raise RuntimeError(FINE_TUNING_NOT_AVAILABLE_MESSAGE)
        self.model_name = str(self.adapter_path)
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Dependances fine-tuning manquantes. "
                "`requirements_finetuning.txt` est requis avant C6/C7."
            ) from exc

        adapter_config_path = self.adapter_path / "adapter_config.json"
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_model_name = adapter_config.get("base_model_name_or_path")
        if not base_model_name:
            raise RuntimeError(
                "adapter_config.json ne contient pas base_model_name_or_path."
            )
        self.base_model_name = str(base_model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.adapter_path,
                local_files_only=False,
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                local_files_only=False,
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                device_map="auto",
                torch_dtype=dtype,
                local_files_only=False,
            )
            self.model = PeftModel.from_pretrained(
                base_model,
                self.adapter_path,
                local_files_only=True,
            )
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(
                "Impossible de charger le modele fine-tune C6/C7. "
                "Le modele de base ou l'adaptateur LoRA est indisponible "
                "ou incompatible."
            ) from exc

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        do_sample = temperature > 0
        generation_kwargs = {
            **inputs,
            "max_new_tokens": settings.ollama_num_predict,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = max(temperature, 1e-5)
            generation_kwargs["top_p"] = 0.9
        with torch.no_grad():
            output_ids = self.model.generate(**generation_kwargs)
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip() or "{}"
