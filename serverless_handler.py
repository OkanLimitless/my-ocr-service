"""RunPod serverless entry point for OCR tasks.

Expects the API to POST:

{
  "input": {
    "task": "tasks.ocr_document" | "tasks.ocr_page" | "tasks.ocr_pdf_document",
    "payload": { ...original Celery kwargs... }
  }
}

Returns a JSON object containing the task result or an error message.
"""

from __future__ import annotations

import logging
import os
import traceback
from typing import Any, Callable

log_level_name = os.getenv("RUNPOD_LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(level=log_level)
logger = logging.getLogger("serverless")
logger.info("Log level set to %s", logging.getLevelName(logger.level))

logger.info("Importing Celery task definitions for serverless dispatch...")

from tasks import ocr_document, ocr_page, ocr_pdf_document

try:  # Imported lazily so unit tests can stub serverless.start
    from runpod import serverless
except Exception:  # pragma: no cover - serverless runner may import handler directly
    serverless = None  # type: ignore


TASKS: dict[str, Callable[..., Any]] = {
    "tasks.ocr_document": ocr_document.run,
    "tasks.ocr_page": ocr_page.run,
    "tasks.ocr_pdf_document": ocr_pdf_document.run,
}


def handler(event: dict[str, Any] | None) -> dict[str, Any]:
    event = event or {}
    input_payload = event.get("input") or {}
    task_name = input_payload.get("task")
    payload = input_payload.get("payload") or {}

    fn = TASKS.get(task_name or "")
    if fn is None:
        logger.warning("Unknown task %s", task_name)
        return {"status": "error", "error": f"unknown task {task_name}"}

    try:
        result = fn(**payload)
        logger.info("Task %s completed", task_name)
        return {"status": "ok", "result": result}
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        traceback.print_exc()
        logger.exception("Task %s failed", task_name)
        return {"status": "error", "error": str(exc)}

if __name__ == "__main__" and serverless is not None:
    if os.getenv("RUNPOD_DISABLE_SERVERLESS", "0") == "1":
        logger.info("RUNPOD_DISABLE_SERVERLESS=1; skipping serverless.start")
    else:
        logger.info("Starting RunPod serverless loop...")
        serverless.start({"handler": handler})
