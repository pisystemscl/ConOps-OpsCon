from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from src.core.schema import ConOpsAnalysis
from src.domain.rule_based_evaluator import evaluate_document_rules


@dataclass(frozen=True)
class DocumentSection:
    section_id: str
    title: str
    start_page: int | None
    end_page: int | None
    text: str


SECTION_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+){0,4}\s+)?[A-Z][A-Za-z0-9,()&/\- ]{6,100}$"
)


def split_document_sections(document_text: str) -> list[DocumentSection]:
    sections: list[DocumentSection] = []
    current_page: int | None = None
    current_title = "Document opening"
    current_lines: list[str] = []
    start_page: int | None = None

    def flush() -> None:
        nonlocal current_lines, current_title, start_page
        text = "\n".join(current_lines).strip()
        if text:
            pages = [
                int(match.group(1))
                for match in re.finditer(r"\[Page\s+(\d+)\]", text)
            ]
            sections.append(
                DocumentSection(
                    section_id=f"SEC-{len(sections) + 1:03d}",
                    title=current_title,
                    start_page=start_page or (min(pages) if pages else None),
                    end_page=max(pages) if pages else current_page,
                    text=text,
                )
            )
        current_lines = []

    for raw_line in document_text.splitlines():
        line = raw_line.strip()
        page_match = re.match(r"\[Page\s+(\d+)\]", line)
        if page_match:
            current_page = int(page_match.group(1))
            if start_page is None:
                start_page = current_page
            current_lines.append(line)
            continue
        if (
            SECTION_HEADING_RE.match(line)
            and len(current_lines) > 8
            and not line.endswith(".")
        ):
            flush()
            current_title = line
            start_page = current_page
        current_lines.append(raw_line)
    flush()
    return sections or [
        DocumentSection("SEC-001", "Document", None, None, document_text)
    ]


def classify_sections(document_text: str) -> pd.DataFrame:
    rows: list[dict] = []
    for section in split_document_sections(document_text):
        analysis = ConOpsAnalysis(raw_notes={"source_text": section.text})
        result = evaluate_document_rules(section.text, analysis)
        assessments = result["rule_assessment_detailed"]
        present = [
            row["rule_id"] for row in assessments if row["status"] == "present"
        ]
        missing = [
            row["rule_id"]
            for row in assessments
            if row["status"] == "missing" and row["expected"] == "mandatory"
        ]
        evidence = [
            row["evidence_quote"]
            for row in assessments
            if row.get("evidence_quote")
        ]
        rows.append(
            {
                "section_id": section.section_id,
                "section_title": section.title,
                "start_page": section.start_page,
                "end_page": section.end_page,
                "classification": result["classification_by_rules"],
                "conops_score": result["conops_rule_score"],
                "opscon_score": result["opscon_rule_score"],
                "rules_present": ", ".join(present),
                "rules_absent": ", ".join(missing),
                "evidence": " | ".join(evidence[:5]),
            }
        )
    return pd.DataFrame(rows)


def export_section_classification(
    document_text: str,
    output_path,
) -> pd.DataFrame:
    frame = classify_sections(document_text)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return frame
