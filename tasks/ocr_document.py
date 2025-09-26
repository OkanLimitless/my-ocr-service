from __future__ import annotations

import os
from typing import Any, Dict

from app import run_ocr_from_bytes

from .utils import PayloadError, decode_binary


def run(**payload: Any) -> Dict[str, Any]:
    """Run OCR over a single image-like document."""

    data_bytes, suffix = decode_binary(payload, default_suffix=payload.get("suffix", ".png"))
    texts = run_ocr_from_bytes(data_bytes, suffix=suffix)

    result: Dict[str, Any] = {"text": texts}
    if payload.get("include_metadata"):
        result["meta"] = {
            "suffix": suffix,
            "size_bytes": len(data_bytes),
            "use_gpu": os.getenv("PADDLEOCR_USE_GPU", "1") != "0",
        }
    return result


__all__ = ["run"]
