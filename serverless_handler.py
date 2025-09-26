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
from typing import Any, Callable, Dict

from runpod import serverless

from tasks import ocr_document, ocr_page, ocr_pdf_document

logger = logging.getLogger("my_ocr_service.serverless")

TASKS: Dict[str, Callable[..., Any]] = {
    "tasks.ocr_document": ocr_document.run,
    "tasks.ocr_page": ocr_page.run,
    "tasks.ocr_pdf_document": ocr_pdf_document.run,
}


def handler(event: Dict[str, Any] | None) -> Dict[str, Any]:
    event = event or {}
    input_payload = event.get("input") or {}
    task_name = input_payload.get("task")
    payload = input_payload.get("payload") or {}

    fn = TASKS.get(task_name or "")
    if fn is None:
        logger.warning("Unknown task received: %s", task_name)
        return {"status": "error", "error": f"unknown task {task_name}"}

    try:
        result = fn(**payload)
        return {"status": "ok", "result": result}
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        traceback.print_exc()
        logger.exception("Task %s failed", task_name)
        return {"status": "error", "error": str(exc)}


def start_serverless() -> None:
    log_level_name = os.getenv("RUNPOD_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.basicConfig(level=log_level)
    serverless.start({"handler": handler})


if os.getenv("RUNPOD_DISABLE_SERVERLESS", "0") != "1":
    start_serverless()
