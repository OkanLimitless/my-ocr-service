from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict

import requests
from runpod import serverless

from app import run_ocr_from_bytes

logger = logging.getLogger("my_ocr_service.serverless")


class HandlerError(Exception):
    """Raised when the incoming event payload is invalid."""


def ocr_document(**payload: Any) -> Dict[str, Any]:
    file_bytes, suffix = _resolve_payload(payload)
    texts = run_ocr_from_bytes(file_bytes, suffix=suffix)
    return {"text": texts}


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    input_payload = event.get("input") or {}
    task = input_payload.get("task")
    payload = input_payload.get("payload") or {}

    try:
        if task == "tasks.ocr_document":
            return {"status": "success", "output": ocr_document(**payload)}
        logger.warning("Received unknown task: %s", task)
        return {"status": "error", "error": "unknown task"}
    except HandlerError as err:
        logger.warning("Invalid payload: %s", err)
        return {"status": "error", "error": str(err)}
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unhandled exception while processing task: %s", task)
        return {"status": "error", "error": str(exc)}


def _resolve_payload(payload: Dict[str, Any]) -> tuple[bytes, str]:
    if "base64" in payload:
        try:
            file_bytes = base64.b64decode(payload["base64"], validate=True)
        except Exception as exc:  # pragma: no cover - invalid base64 path
            raise HandlerError("Failed to decode base64 input") from exc
        suffix = payload.get("suffix") or payload.get("filename")
        return file_bytes, _suffix_from_name(suffix)

    if "url" in payload:
        url = payload["url"]
        timeout = float(os.getenv("RUNPOD_HTTP_TIMEOUT", "30"))
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
        except Exception as exc:
            raise HandlerError(f"Failed to download file from URL: {url}") from exc
        suffix = payload.get("suffix") or Path(url).suffix
        return response.content, _suffix_from_name(suffix)

    raise HandlerError("Payload must include 'base64' or 'url'.")


def _suffix_from_name(name: str | None) -> str:
    if not name:
        return ".png"
    suffix = Path(name).suffix
    return suffix if suffix else ".png"


def start_serverless() -> None:
    log_level_name = os.getenv("RUNPOD_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.basicConfig(level=log_level)
    serverless.start({"handler": handler})


if os.getenv("RUNPOD_DISABLE_SERVERLESS", "0") != "1":
    start_serverless()
