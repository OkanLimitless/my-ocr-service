from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("my_ocr_service")

app = FastAPI(title="PaddleOCR Service", version="1.0.0")

ocr_model: PaddleOCR | None = None

CHUNK_SIZE = 1 * 1024 * 1024  # 1 MiB
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MiB
SUPPORTED_CONTENT_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/tiff",
    "image/bmp",
    "image/gif",
}


@app.on_event("startup")
async def load_model() -> None:
    global ocr_model
    logger.info("Loading PaddleOCR model (lang=en, use_angle_cls=True)...")
    try:
        ocr_model = PaddleOCR(lang="en", use_angle_cls=True, show_log=False)
        logger.info("PaddleOCR model loaded successfully.")
    except Exception as exc:  # pragma: no cover - startup failure path
        logger.exception("Failed to initialize PaddleOCR model.")
        raise RuntimeError("PaddleOCR initialization failed") from exc


@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)) -> JSONResponse:
    if ocr_model is None:
        logger.error("OCR model is not available when handling a request.")
        raise HTTPException(status_code=503, detail="OCR service not ready yet. Try again later.")

    if file.content_type and not (
        file.content_type in SUPPORTED_CONTENT_TYPES or file.content_type.startswith("image/")
    ):
        raise HTTPException(status_code=400, detail="Unsupported content type.")

    temp_path = ""
    try:
        temp_path = await _persist_upload(file)
        logger.info("Saved upload to %s", temp_path)
        ocr_result = await asyncio.to_thread(_run_ocr, temp_path)
        texts = _extract_text(ocr_result)
        return JSONResponse({"text": texts})
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("OCR processing failed.")
        raise HTTPException(status_code=500, detail="Failed to process the uploaded file.") from exc
    finally:
        _cleanup_temp_file(temp_path)
        await _close_upload(file)


def _run_ocr(temp_path: str) -> List[List]:
    if ocr_model is None:  # Should not happen; guard to satisfy type checker.
        raise RuntimeError("OCR model is not initialized.")
    return ocr_model.ocr(temp_path, cls=True)


async def _persist_upload(upload_file: UploadFile) -> str:
    suffix = Path(upload_file.filename or "").suffix
    temp_file = None
    total_bytes = 0

    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        while True:
            chunk = await upload_file.read(CHUNK_SIZE)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_SIZE:
                raise HTTPException(status_code=413, detail="Uploaded file is too large.")
            temp_file.write(chunk)

        temp_file.flush()
        temp_file.close()

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        return temp_file.name
    except Exception:
        if temp_file is not None:
            try:
                temp_file.close()
            finally:
                _cleanup_temp_file(temp_file.name)
        raise


async def _close_upload(upload_file: UploadFile) -> None:
    try:
        await upload_file.close()
    except Exception:  # pragma: no cover - best effort cleanup
        logger.warning("Failed to close uploaded file stream.", exc_info=True)


def _cleanup_temp_file(path: str) -> None:
    if not path:
        return
    try:
        os.remove(path)
        logger.info("Deleted temporary file %s", path)
    except FileNotFoundError:
        pass
    except Exception:
        logger.warning("Failed to remove temporary file %s", path, exc_info=True)


def _extract_text(ocr_result: List) -> List[str]:
    texts: List[str] = []
    if not ocr_result:
        return texts

    for page in ocr_result:
        if not page:
            continue
        for line in page:
            if not line or len(line) < 2:
                continue
            text_block = line[1]
            if not text_block:
                continue
            text = text_block[0] if isinstance(text_block, (list, tuple)) else text_block
            if text:
                texts.append(str(text))
    return texts
