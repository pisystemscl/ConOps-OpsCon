from __future__ import annotations
from dataclasses import dataclass
from .pdf_loader import PageText

@dataclass
class Chunk:
    id: str
    source: str
    page: int | None
    text: str
    section: str = ""
    source_path: str = ""


def _guess_section(text: str) -> str:
    for line in text.splitlines():
        candidate = line.strip()
        if 3 <= len(candidate) <= 120:
            return candidate
    return ""


def split_text(text: str, chunk_size: int = 1200, overlap: int = 180, source: str = "document", page: int | None = None, source_path: str = "") -> list[Chunk]:
    if not text:
        return []
    words = text.split()
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        if len(chunk_text.strip()) > 50:
            chunks.append(
                Chunk(
                    id=f"{source}_p{page or 0}_{idx}",
                    source=source,
                    page=page,
                    text=chunk_text,
                    section=_guess_section(text),
                    source_path=source_path,
                )
            )
        if end == len(words):
            break
        start = max(0, end - overlap)
        idx += 1
    return chunks


def split_pages(pages: list[PageText], chunk_size: int = 1200, overlap: int = 180) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for page in pages:
        all_chunks.extend(
            split_text(
                page.text,
                chunk_size,
                overlap,
                page.source,
                page.page,
                source_path=getattr(page, "source_path", ""),
            )
        )
    return all_chunks
