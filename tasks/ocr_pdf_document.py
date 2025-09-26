from __future__ import annotations

import os
from typing import Any, Dict, List

from app import run_ocr_from_bytes

from .utils import DEFAULT_DPI, PayloadError, decode_binary, render_pdf_pages


def run(**payload: Any) -> Dict[str, Any]:
    """Run OCR over an entire PDF document."""

    pdf_bytes, _ = decode_binary(payload, default_suffix=payload.get("suffix", ".pdf"))
    dpi = int(payload.get("dpi") or os.getenv("PADDLEOCR_PDF_DPI", DEFAULT_DPI))
    page_indices = payload.get("page_indices")

    pages: List[Dict[str, Any]] = []
    for index, image_bytes in render_pdf_pages(pdf_bytes, page_indices=page_indices, dpi=dpi):
        text = run_ocr_from_bytes(image_bytes, suffix=".png")
        pages.append({"page_index": index, "text": text})

    if not pages:
        raise PayloadError("No pages were rendered from the provided PDF.")

    return {"pages": pages}


__all__ = ["run"]
