from __future__ import annotations

import os
from typing import Any, Dict

from app import run_ocr_from_bytes

from .utils import DEFAULT_DPI, PayloadError, decode_binary, render_pdf_pages


def run(**payload: Any) -> Dict[str, Any]:
    """Render a single PDF page and run OCR on it."""

    page_index = payload.get("page_index")
    if page_index is None:
        raise PayloadError("'page_index' is required for tasks.ocr_page")

    pdf_bytes, _ = decode_binary(payload, default_suffix=payload.get("suffix", ".pdf"))
    dpi = int(payload.get("dpi") or os.getenv("PADDLEOCR_PDF_DPI", DEFAULT_DPI))

    rendered_pages = list(render_pdf_pages(pdf_bytes, page_indices=int(page_index), dpi=dpi))
    if not rendered_pages:
        raise PayloadError("Requested page could not be rendered.")

    index, image_bytes = rendered_pages[0]
    text = run_ocr_from_bytes(image_bytes, suffix=".png")
    return {
        "page_index": index,
        "text": text,
    }


__all__ = ["run"]
