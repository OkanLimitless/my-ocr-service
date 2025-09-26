from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import fitz  # type: ignore
import requests

DEFAULT_DPI = int(os.getenv("PADDLEOCR_PDF_DPI", "180"))
DEFAULT_SUFFIX = ".png"


class PayloadError(ValueError):
    """Raised when the payload does not contain the expected fields."""


def decode_binary(payload: dict, *, default_suffix: str = DEFAULT_SUFFIX) -> Tuple[bytes, str]:
    if not isinstance(payload, dict):
        raise PayloadError("Payload must be a mapping.")

    if "data" in payload and isinstance(payload["data"], (bytes, bytearray)):
        suffix = _suffix_from_payload(payload, default_suffix)
        return bytes(payload["data"]), suffix

    if "base64" in payload:
        try:
            data_bytes = base64.b64decode(payload["base64"], validate=True)
        except Exception as exc:
            raise PayloadError("Failed to decode base64 input") from exc
        suffix = _suffix_from_payload(payload, default_suffix)
        return data_bytes, suffix

    if "url" in payload:
        url = payload["url"]
        timeout = float(os.getenv("RUNPOD_HTTP_TIMEOUT", "30"))
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
        except Exception as exc:
            raise PayloadError(f"Failed to download file from URL: {url}") from exc
        derived_payload = dict(payload)
        derived_payload.setdefault("filename", url)
        suffix = _suffix_from_payload(derived_payload, default_suffix)
        return response.content, suffix

    raise PayloadError("Payload must include 'data', 'base64', or 'url'.")


def render_pdf_pages(
    pdf_bytes: bytes,
    *,
    page_indices: Sequence[int] | int | None = None,
    dpi: int | None = None,
) -> Iterable[Tuple[int, bytes]]:
    dpi_value = dpi or DEFAULT_DPI
    scale = dpi_value / 72.0

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        total_pages = document.page_count
        indices = _normalize_page_indices(page_indices, total_pages)

        for index in indices:
            page = document.load_page(index)
            matrix = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=matrix)
            yield index, pix.tobytes("png")


def _normalize_page_indices(page_indices: Sequence[int] | int | None, total: int) -> List[int]:
    if page_indices is None:
        return list(range(total))

    if isinstance(page_indices, int):
        indices = [page_indices]
    elif isinstance(page_indices, (list, tuple, set)):
        try:
            indices = [int(idx) for idx in page_indices]
        except Exception as exc:
            raise PayloadError("page_indices must contain integers.") from exc
    else:
        raise PayloadError("page_indices must be an int or a sequence of ints.")

    if not indices:
        raise PayloadError("page_indices cannot be empty.")

    unique_sorted = sorted(set(indices))
    for idx in unique_sorted:
        if idx < 0 or idx >= total:
            raise PayloadError(f"page index {idx} out of range; total pages: {total}.")
    return unique_sorted


def _suffix_from_payload(payload: dict, default_suffix: str) -> str:
    candidates = [payload.get("suffix"), payload.get("filename"), payload.get("name")]
    for candidate in candidates:
        if candidate:
            suffix = Path(str(candidate)).suffix
            if suffix:
                return suffix
    return default_suffix


__all__ = [
    "PayloadError",
    "decode_binary",
    "render_pdf_pages",
]
