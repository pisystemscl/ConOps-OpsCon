from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pypdf import PdfReader


OCRProgress = Callable[[int, int, str], None]
OCR_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "ocr_cache"


@dataclass
class PageText:
    source: str
    page: int
    text: str
    source_path: str = ""


def _clean_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _native_pdf_pages(path: Path) -> tuple[list[PageText], int]:
    reader = PdfReader(str(path))
    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = _clean_text(page.extract_text() or "")
        except Exception:
            text = ""
        if text:
            pages.append(
                PageText(
                    source=path.name,
                    page=index,
                    text=text,
                    source_path=str(path.resolve()),
                )
            )
    return pages, len(reader.pages)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_page_indices(total_pages: int, maximum: int) -> list[int]:
    if total_pages <= maximum:
        return list(range(total_pages))
    if maximum <= 1:
        return [0]
    return sorted(
        {
            round(index * (total_pages - 1) / (maximum - 1))
            for index in range(maximum)
        }
    )


def _prioritize_page_indices(indices: list[int]) -> list[int]:
    """Order sampled pages to maximize document coverage early."""
    remaining = sorted(set(indices))
    if len(remaining) <= 2:
        return remaining
    selected = [remaining.pop(0), remaining.pop(-1)]
    while remaining:
        candidate = max(
            remaining,
            key=lambda value: min(abs(value - item) for item in selected),
        )
        selected.append(candidate)
        remaining.remove(candidate)
    return selected


def _ocr_cache_path(
    path: Path,
    *,
    max_pages: int,
    target_chars: int,
    scale: float,
) -> Path:
    settings_key = f"v2-{max_pages}-{target_chars}-{scale:.2f}"
    return OCR_CACHE_DIR / f"{_file_hash(path)}-{settings_key}.json"


def _load_ocr_cache(
    path: Path,
    *,
    max_pages: int,
    target_chars: int,
    scale: float,
) -> list[PageText] | None:
    cache_path = _ocr_cache_path(
        path,
        max_pages=max_pages,
        target_chars=target_chars,
        scale=scale,
    )
    if not cache_path.exists():
        return None
    try:
        rows = json.loads(cache_path.read_text(encoding="utf-8"))
        return [
            PageText(
                source=path.name,
                page=int(row["page"]),
                text=str(row["text"]),
                source_path=str(path.resolve()),
            )
            for row in rows
            if row.get("text")
        ]
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _save_ocr_cache(
    path: Path,
    pages: list[PageText],
    *,
    max_pages: int,
    target_chars: int,
    scale: float,
) -> None:
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _ocr_cache_path(
        path,
        max_pages=max_pages,
        target_chars=target_chars,
        scale=scale,
    )
    cache_path.write_text(
        json.dumps(
            [
                {"page": page.page, "text": page.text}
                for page in pages
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _ocr_pdf_pages(
    path: Path,
    *,
    max_pages: int,
    target_chars: int,
    scale: float,
    progress_callback: OCRProgress | None,
) -> list[PageText]:
    try:
        import fitz
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "Ce PDF est scanne et necessite les dependances OCR. "
            "Commande associee: pip install -r requirements_ocr.txt"
        ) from exc

    cached = _load_ocr_cache(
        path,
        max_pages=max_pages,
        target_chars=target_chars,
        scale=scale,
    )
    if cached is not None:
        if progress_callback:
            progress_callback(
                len(cached),
                len(cached),
                "Texte OCR charge depuis le cache local.",
            )
        return cached

    document = fitz.open(str(path))
    selected_indices = _prioritize_page_indices(
        _sample_page_indices(
            document.page_count,
            min(max_pages, document.page_count),
        )
    )
    engine = RapidOCR()
    pages: list[PageText] = []
    extracted_chars = 0
    total = len(selected_indices)

    for position, page_index in enumerate(selected_indices, start=1):
        if progress_callback:
            progress_callback(
                position - 1,
                total,
                f"OCR de la page {page_index + 1}/{document.page_count}",
            )
        page = document[page_index]
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            alpha=False,
        )
        image = np.frombuffer(
            pixmap.samples,
            dtype=np.uint8,
        ).reshape(pixmap.height, pixmap.width, pixmap.n)
        result, _ = engine(image)
        text = _clean_text(
            "\n".join(item[1] for item in (result or []) if item[1])
        )
        if text:
            pages.append(
                PageText(
                    source=path.name,
                    page=page_index + 1,
                    text=text,
                    source_path=str(path.resolve()),
                )
            )
            extracted_chars += len(text)
        if extracted_chars >= target_chars:
            break

    document.close()
    if progress_callback:
        progress_callback(
            len(pages),
            len(pages),
            f"OCR termine: {len(pages)} pages, {extracted_chars} caracteres.",
        )
    if pages:
        _save_ocr_cache(
            path,
            pages,
            max_pages=max_pages,
            target_chars=target_chars,
            scale=scale,
        )
    return pages


def extract_pdf_pages(
    pdf_path: str | Path,
    *,
    enable_ocr: bool = True,
    ocr_max_pages: int | None = None,
    ocr_target_chars: int | None = None,
    ocr_scale: float | None = None,
    progress_callback: OCRProgress | None = None,
) -> list[PageText]:
    path = Path(pdf_path)
    native_pages, total_pages = _native_pdf_pages(path)
    native_chars = sum(len(page.text) for page in native_pages)
    if native_chars >= 200 or not enable_ocr:
        return native_pages

    from src.config import settings

    if progress_callback:
        progress_callback(
            0,
            min(ocr_max_pages or settings.ocr_max_pages, total_pages),
            "Aucun texte natif detecte. Demarrage de l'OCR local.",
        )
    return _ocr_pdf_pages(
        path,
        max_pages=ocr_max_pages or settings.ocr_max_pages,
        target_chars=ocr_target_chars or settings.ocr_target_chars,
        scale=ocr_scale or settings.ocr_render_scale,
        progress_callback=progress_callback,
    )


def extract_pdf_text(
    pdf_path: str | Path,
    **kwargs,
) -> str:
    pages = extract_pdf_pages(pdf_path, **kwargs)
    return "\n\n".join(f"[Page {page.page}]\n{page.text}" for page in pages)


def load_pdfs_from_folder(
    folder: str | Path,
    *,
    enable_ocr: bool = True,
) -> list[PageText]:
    folder = Path(folder)
    all_pages: list[PageText] = []
    for pdf in sorted(folder.glob("*.pdf")):
        all_pages.extend(extract_pdf_pages(pdf, enable_ocr=enable_ocr))
    return all_pages
