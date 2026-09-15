from __future__ import annotations
import json
from src.core.schema import ConOpsAnalysis
from src.llm.providers import BaseLLM
from src.llm.prompts import build_extraction_prompt
from src.core.rag_engine import RagEngine
from src.config import settings
from src.llm.json_guard import JsonGuardError, safe_llm_json_call
from src.pipeline.post_processor import enrich_analysis
from src.knowledge.ontology import get_knowledge_context

class ConOpsExtractor:
    def __init__(self, llm: BaseLLM, rag: RagEngine | None = None):
        self.llm = llm
        self.rag = rag

    def extract(
        self,
        document_text: str,
        *,
        use_template: bool = True,
        use_rules: bool = True,
        use_rag: bool = False,
        configuration_name: str = "C0",
        knowledge_format: str = "K1",
    ) -> ConOpsAnalysis:
        rag_context = ""
        if use_rag and self.rag:
            rag_context = self.rag.context_for_prompt(
                document_text,
                top_k=settings.rag_top_k,
                expand_query=True,
            )
        prompt = build_extraction_prompt(
            document_text,
            use_template=use_template,
            use_rules=use_rules,
            rag_context=rag_context,
            configuration_name=configuration_name,
            knowledge_context=(
                get_knowledge_context(knowledge_format) if use_rules else ""
            ),
        )
        try:
            json_result = safe_llm_json_call(
                prompt,
                ConOpsAnalysis,
                self.llm,
                max_retries=3,
            )
            data = json_result.data
        except JsonGuardError as error:
            wrapped = ValueError(f"JSON LLM invalide: {error}")
            wrapped.raw_response = error.raw_response
            wrapped.repaired_response = error.repaired_response
            wrapped.attempts = error.attempts
            raise wrapped from error
        except ValueError as error:
            raise ValueError(f"JSON LLM invalide: {error}") from error
        analysis = ConOpsAnalysis.model_validate(data)
        analysis.raw_notes["json_valid"] = json_result.json_valid
        analysis.raw_notes["json_repair_attempts"] = json_result.json_repair_attempts
        analysis.raw_notes["json_error_message"] = json_result.json_error_message
        analysis.raw_notes["raw_llm_response"] = json_result.raw_response
        analysis.raw_notes["configuration"] = configuration_name
        analysis.raw_notes["rag_used"] = bool(rag_context)
        analysis.raw_notes["rag_context"] = rag_context
        analysis.raw_notes["llm_model"] = getattr(self.llm, "model", self.llm.__class__.__name__)
        analysis.raw_notes["knowledge_format"] = knowledge_format
        analysis = enrich_analysis(analysis, document_text)
        if use_rag and self.rag:
            traceable_items = []
            groups = (
                ("STK", analysis.stakeholders, "name"),
                ("REQ", analysis.requirements, "text"),
                ("INT", analysis.interfaces, "exchanged_information"),
                ("RISK", analysis.risks, "description"),
            )
            for prefix, items, text_field in groups:
                for index, item in enumerate(items, start=1):
                    item_id = (
                        getattr(item, "id", "")
                        or getattr(item, "interface_id", "")
                        or f"{prefix}-{index:03d}"
                    )
                    traceable_items.append(
                        (
                            item_id,
                            str(getattr(item, text_field, "")).lower(),
                            [
                                evidence.text.lower()
                                for evidence in item.evidence
                                if evidence.text
                            ],
                        )
                    )
            for row in self.rag.retrieval_log:
                chunk_text = row["chunk_text"].lower()
                chunk_tokens = set(chunk_text.split())
                related_item_ids = []
                for item_id, item_text, evidence_quotes in traceable_items:
                    item_tokens = set(item_text.split())
                    lexical_overlap = (
                        len(chunk_tokens & item_tokens)
                        / len(item_tokens)
                        if item_tokens
                        else 0.0
                    )
                    evidence_match = any(
                        len(quote) >= 20 and quote[:80] in chunk_text
                        for quote in evidence_quotes
                    )
                    if evidence_match or lexical_overlap >= 0.35:
                        related_item_ids.append(item_id)
                row["related_item_ids"] = related_item_ids
                row["used_in_answer"] = bool(related_item_ids)
            analysis.raw_notes["rag_chunks"] = list(self.rag.retrieval_log)
        return analysis


def analysis_to_compact_text(analysis: ConOpsAnalysis) -> str:
    data = analysis.model_dump()
    return json.dumps(data, ensure_ascii=False, indent=2)
