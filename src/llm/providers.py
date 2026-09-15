from __future__ import annotations
import json
import re
from abc import ABC, abstractmethod
from typing import Any
import requests
from src.config import settings

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        raise NotImplementedError

class MockLLM(BaseLLM):
    """Extraction heuristique déterministe utilisée comme fournisseur local."""

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        if '"verdict"' in prompt and 'hallucination_level' in prompt:
            return json.dumps({
                "verdict": "Bon",
                "recommendations": [
                    "Renforcer la traçabilité des exigences vers les passages du ConOps.",
                    "Séparer explicitement la vision stratégique ConOps des scénarios opérationnels détaillés OpsCon.",
                    "Ajouter une matrice As-Is / To-Be / Gaps / Capacités."
                ],
                "hallucination_level": "faible",
                "comment": "Évaluation heuristique locale."
            }, ensure_ascii=False)
        text = prompt[-24000:]
        return json.dumps(self._heuristic_analysis(text), ensure_ascii=False, indent=2)

    def _sentences(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text)
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if 40 <= len(p.strip()) <= 320]

    def _find(self, sentences: list[str], keywords: list[str], limit: int = 6) -> list[str]:
        out = []
        for s in sentences:
            low = s.lower()
            if any(k.lower() in low for k in keywords):
                out.append(s)
            if len(out) >= limit:
                break
        return out

    def _stakeholders(self, text: str) -> list[dict[str, Any]]:
        candidates = [
            "Government", "Autorité", "Authority", "Regulator", "Operator", "Users", "Airspace Users", "ANSP",
            "Airport Operators", "NASA", "NOAA", "EUROCONTROL", "Service Provider", "UAS operator", "General Public",
            "Passengers", "Emergency Responders", "Air Traffic Control", "Airlines", "Industry", "Stakeholders"
        ]
        found = []
        for c in candidates:
            if re.search(re.escape(c), text, flags=re.IGNORECASE):
                found.append({
                    "name": c,
                    "role": "Partie prenante identifiée dans le document",
                    "interest": "Contribuer ou bénéficier du futur concept d'opérations",
                    "influence_level": "medium",
                    "evidence": []
                })
        return found[:12] or [{"name": "Parties prenantes à identifier", "role": "Non explicite", "interest": "À compléter", "influence_level": "medium", "evidence": []}]

    def _heuristic_analysis(self, text: str) -> dict[str, Any]:
        sentences = self._sentences(text)
        title_match = re.search(r"(?:Concept of Operations|CONOPS|ConOps|Operational Concept)[^\n]{0,120}", text, re.I)
        title = title_match.group(0).strip() if title_match else "ConOps analysé"
        as_is = self._find(sentences, ["current", "existing", "as-is", "legacy", "actuel", "existing systems"], 5)
        to_be = self._find(sentences, ["future", "to-be", "vision", "expected", "2029", "evolution", "target"], 5)
        services = self._find(sentences, ["service", "services", "capability", "capabilities"], 6)
        gaps = self._find(sentences, ["gap", "shortcoming", "challenge", "barrier", "constraint", "problem"], 6)
        risks = self._find(sentences, ["risk", "safety", "security", "cyber", "resilience", "contingency"], 5)
        interfaces = self._find(sentences, ["interface", "information exchange", "data exchange", "interoperability", "integration"], 5)
        reqs = []
        for i, s in enumerate(self._find(sentences, ["shall", "must", "required", "needs to", "requirement", "should"], 8), start=1):
            reqs.append({"id": f"REQ-{i:03d}", "text": s, "priority": "medium", "source_category": "ConOps", "evidence": []})
        if not reqs:
            for i, s in enumerate(services[:4], start=1):
                reqs.append({"id": f"REQ-{i:03d}", "text": "Le futur système doit permettre : " + s, "priority": "medium", "source_category": "inferred_high_level", "evidence": []})
        return {
            "document_title": title,
            "purpose_scope": (sentences[0] if sentences else "Objectif à préciser"),
            "general_context": " ".join(sentences[:3])[:900],
            "as_is": as_is,
            "to_be": to_be,
            "stakeholders": self._stakeholders(text),
            "existing_systems": self._find(sentences, ["system", "systems", "infrastructure", "architecture", "legacy"], 6),
            "existing_services": services[:3],
            "expected_services": services[3:] or services[:3],
            "gaps": gaps,
            "capabilities": self._find(sentences, ["capability", "capacity", "performance", "automation", "scalability"], 6),
            "requirements": reqs,
            "interfaces": [{"source":"Acteur/Système source", "target":"Acteur/Système cible", "exchanged_information":s, "criticality":"medium", "evidence":[]} for s in interfaces],
            "constraints": self._find(sentences, ["constraint", "regulation", "standard", "security", "safety", "environment"], 6),
            "risks": [{"description":s, "impact":"medium", "probability":"medium", "mitigation":"À définir dans l'OpsCon ou l'architecture cible", "evidence":[]} for s in risks],
            "assumptions": ["Les éléments non explicitement cités restent à valider avec les experts métier."],
            "future_actions": [
                {"action":"Compléter la matrice stakeholders / services / capacités", "owner":"Équipe MBSE", "horizon":"court terme", "rationale":"Améliorer la traçabilité du ConOps"},
                {"action":"Valider les exigences haut niveau avec les parties prenantes", "owner":"Experts métier", "horizon":"moyen terme", "rationale":"Réduire les ambiguïtés"},
                {"action":"Transformer les capacités validées en éléments SysML v2", "owner":"Équipe systèmes", "horizon":"moyen terme", "rationale":"Préparer la modélisation formelle"}
            ],
            "things_to_change": gaps[:4],
            "things_to_avoid": ["Mélanger ConOps stratégique et procédures détaillées OpsCon", "Inventer des exigences non traçables"],
            "conops_vs_opscon_comment": "Analyse heuristique : vérifier que le document reste au niveau vision/services/gaps et ne descend pas trop dans les scénarios détaillés.",
            "summary": "Résumé automatique local : le document décrit un concept d'opérations, ses parties prenantes, services attendus, capacités et points de transformation."
        }

class OpenAILLM(BaseLLM):
    def __init__(self, model: str | None = None):
        from openai import OpenAI
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY manquant dans .env")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.openai_model

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or "{}"

class OllamaLLM(BaseLLM):
    def __init__(self, model: str | None = None):
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = model or settings.ollama_model

    def generate(self, prompt: str, temperature: float = 0.1) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": settings.ollama_keep_alive,
            "options": {
                "temperature": temperature,
                "top_p": 0.1,
                "num_ctx": settings.ollama_num_ctx,
                "num_predict": settings.ollama_num_predict,
            },
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=(10, settings.ollama_timeout_seconds),
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise RuntimeError(
                f"Ollama n'a pas répondu dans le délai de "
                f"{settings.ollama_timeout_seconds}s pour le modèle {self.model}."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Échec de l'appel Ollama ({self.model}) sur {self.base_url}: {exc}"
            ) from exc
        return response.json().get("response", "{}")


def list_ollama_models(base_url: str | None = None) -> list[str]:
    """Return locally installed Ollama models without making the UI fail."""
    url = (base_url or settings.ollama_base_url).rstrip("/")
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        return sorted(
            {
                str(item.get("name", "")).strip()
                for item in models
                if isinstance(item, dict) and item.get("name")
            }
        )
    except (requests.RequestException, ValueError, TypeError):
        return []


def get_llm(provider: str | None = None, model: str | None = None) -> BaseLLM:
    provider = (provider or settings.llm_provider).lower()
    if provider == "openai":
        return OpenAILLM(model=model)
    if provider == "ollama":
        return OllamaLLM(model=model)
    if provider == "mock":
        return MockLLM()
    raise ValueError(f"Provider LLM inconnu : {provider!r}. Valeurs acceptées : mock, ollama, openai.")


def get_llm_for_configuration(
    code: str,
    provider: str | None = None,
    model: str | None = None,
) -> BaseLLM:
    provider = (provider or settings.llm_provider).lower()
    if code in {"C6", "C7"}:
        from src.llm.fine_tuned_provider import FineTunedLLM

        return FineTunedLLM(adapter_path=model)
    model_map = {
        "C1": settings.ollama_model_c1,
        "C2": settings.ollama_model_c2,
        "C3": settings.ollama_model_c3,
        "C4": settings.ollama_model_c4,
        "C5": settings.ollama_model_c5,
        "C6": settings.ollama_model_c6,
        "C7": settings.ollama_model_c7,
    }
    if provider == "ollama":
        return OllamaLLM(model=model or model_map.get(code) or settings.ollama_model)
    if provider == "openai":
        return OpenAILLM(model=model or settings.openai_model)
    if provider == "mock":
        return MockLLM()
    raise ValueError(f"Provider LLM inconnu : {provider!r}. Valeurs acceptées : mock, ollama, openai.")
