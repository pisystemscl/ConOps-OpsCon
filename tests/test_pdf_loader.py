from pathlib import Path

from src.core.pdf_loader import (
    _ocr_cache_path,
    _prioritize_page_indices,
    _sample_page_indices,
)


def test_ocr_page_sampling_covers_beginning_middle_and_end():
    indices = _sample_page_indices(total_pages=105, maximum=5)

    assert indices[0] == 0
    assert indices[-1] == 104
    assert 50 <= indices[len(indices) // 2] <= 54
    assert len(indices) == 5


def test_ocr_cache_key_depends_on_extraction_settings(tmp_path):
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"fake pdf content")

    first = _ocr_cache_path(
        pdf_path,
        max_pages=20,
        target_chars=40000,
        scale=1.0,
    )
    second = _ocr_cache_path(
        pdf_path,
        max_pages=40,
        target_chars=80000,
        scale=1.15,
    )

    assert first != second


def test_ocr_priority_covers_document_extremes_and_middle_first():
    sampled = _sample_page_indices(total_pages=105, maximum=9)
    prioritized = _prioritize_page_indices(sampled)

    assert prioritized[:3] == [0, 104, 52]
    assert sorted(prioritized) == sampled
