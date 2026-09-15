from src.pipeline.experiment_runner import CONFIGURATIONS
from src.llm.providers import BaseLLM, get_llm_for_configuration


def test_core_configuration_contracts():
    configurations = {item.code: item for item in CONFIGURATIONS}

    assert configurations["C1"].use_rag is False
    assert configurations["C1"].use_rules is False
    assert configurations["C1"].use_llm_evaluator is False

    assert configurations["C3"].use_rag is False
    assert configurations["C3"].use_rules is True

    assert configurations["C4"].use_rag is True
    assert configurations["C4"].use_rules is False

    assert configurations["C5"].use_rag is True
    assert configurations["C5"].use_rules is True
    assert configurations["C5"].use_llm_evaluator is True


def test_c6_c7_use_fine_tuned_provider(monkeypatch):
    class FakeFineTunedLLM(BaseLLM):
        def __init__(self, adapter_path=None):
            self.adapter_path = adapter_path

        def generate(self, prompt: str, temperature: float = 0.1) -> str:
            return "{}"

    monkeypatch.setattr(
        "src.llm.fine_tuned_provider.FineTunedLLM",
        FakeFineTunedLLM,
    )

    c6 = get_llm_for_configuration("C6", provider="mock", model="models/conops-lora")
    c7 = get_llm_for_configuration("C7", provider="ollama", model="models/conops-lora")

    assert isinstance(c6, FakeFineTunedLLM)
    assert isinstance(c7, FakeFineTunedLLM)
    assert c6.adapter_path == "models/conops-lora"
