from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("my_ocr_service")

try:
    from paddleocr import PaddleOCR
except Exception as import_exc:  # pragma: no cover - import error path
    logger.exception(
        "Failed to import PaddleOCR. Verify GPU drivers and paddlepaddle-gpu wheel compatibility."  # noqa: E501
    )
    raise

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="PaddleOCR Service", version="1.0.0")

ocr_model: PaddleOCR | None = None
OCR_INIT_TRIED = False

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


@app.get("/")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)) -> JSONResponse:
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


def get_model() -> PaddleOCR:
    global ocr_model, OCR_INIT_TRIED

    if ocr_model is not None:
        return ocr_model

    if OCR_INIT_TRIED:
        raise RuntimeError("Previous PaddleOCR initialization failed. Inspect logs for details.")

    OCR_INIT_TRIED = True
    use_gpu = os.getenv("PADDLEOCR_USE_GPU", "1") != "0"

    init_kwargs = {"lang": "en", "use_angle_cls": True, "show_log": False, "use_gpu": use_gpu}

    try:
        logger.info("Initializing PaddleOCR (use_gpu=%s)...", use_gpu)
        ocr_model_local = PaddleOCR(**init_kwargs)
        logger.info("PaddleOCR model loaded successfully.")
        ocr_model = ocr_model_local
        return ocr_model_local
    except Exception as exc:
        logger.warning("PaddleOCR GPU initialization failed: %s", exc)
        if use_gpu:
            try:
                logger.info("Retrying PaddleOCR initialization on CPU...")
                init_kwargs["use_gpu"] = False
                ocr_model_local = PaddleOCR(**init_kwargs)
                logger.info("PaddleOCR CPU model loaded successfully.")
                ocr_model = ocr_model_local
                return ocr_model_local
            except Exception:
                logger.exception("PaddleOCR CPU fallback also failed.")
                raise
        raise


def _run_ocr(temp_path: str) -> List[List]:
    model = get_model()
    return model.ocr(temp_path, cls=True)


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


def run_ocr_from_bytes(data: bytes, suffix: str = ".png") -> List[str]:
    if not data:
        raise ValueError("No data provided for OCR.")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temp_file.write(data)
        temp_file.flush()
        temp_file.close()
        return _extract_text(_run_ocr(temp_file.name))
    finally:
        _cleanup_temp_file(temp_file.name)
