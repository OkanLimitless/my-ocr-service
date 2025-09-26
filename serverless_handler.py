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

import traceback
from typing import Any, Callable

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
        return {"status": "error", "error": f"unknown task {task_name}"}

    try:
        result = fn(**payload)
        return {"status": "ok", "result": result}
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        traceback.print_exc()
        return {"status": "error", "error": str(exc)}


if __name__ == "__main__" and serverless is not None:
    serverless.start({"handler": handler})
