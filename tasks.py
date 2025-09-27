import asyncio
import io
import json
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Tuple

import boto3
import cv2
import numpy as np
import pypdfium2 as pdfium

from celery_app import celery_app


logger = logging.getLogger("ocr_tasks")

if os.getenv("DATABASE_URL"):
    logger.info("DATABASE_URL detected at startup")
else:
    logger.info("DATABASE_URL not provided; database writes disabled")


def _run_sync(coro):
    """Execute an async coroutine even when an event loop is already running."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_holder: dict[str, object] = {}

    def _runner() -> None:
        try:
            result_holder["result"] = asyncio.run(coro)
        except Exception as exc:  # pragma: no cover - diagnostic path
            result_holder["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in result_holder:
        raise result_holder["error"]  # type: ignore[misc]
    return result_holder.get("result")


def s3_client():
    endpoint = os.getenv("S3_ENDPOINT_URL")
    region = os.getenv("S3_REGION")
    key_id = os.getenv("S3_ACCESS_KEY_ID")
    secret = os.getenv("S3_SECRET_ACCESS_KEY")
    session = boto3.session.Session()
    client = session.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
    )
    logger.debug(
        "Initialized S3 client",
        extra={
            "endpoint": endpoint,
            "region": region,
            "bucket": os.getenv("S3_BUCKET"),
        },
    )
    return client


def _canonicalize_db_url(url: str) -> str:
    if not url or not url.startswith("postgres"):
        return url
    try:
        scheme, rest = url.split("://", 1)
    except ValueError:
        return url
    if "@" not in rest:
        return url
    userinfo, hostpart = rest.split("@", 1)
    if ":" in userinfo:
        user, pwd = userinfo.split(":", 1)
    else:
        user, pwd = userinfo, ""
    from urllib.parse import quote
    user_q = quote(user, safe="") if user else ""
    pwd_q = quote(pwd, safe="") if pwd else ""
    return f"{scheme}://{user_q}:{pwd_q}@{hostpart}"


async def _db_exec(query: str, *args):
    import asyncpg  # lazy import

    dsn = _canonicalize_db_url(os.getenv("DATABASE_URL", ""))
    if not dsn:
        logger.debug("DATABASE_URL not set; skipping query", extra={"query": query})
        return None
    try:
        conn = await asyncpg.connect(dsn)
        logger.debug("Connected to PostgreSQL", extra={"query": query.split()[0] if query else ""})
        try:
            result = await conn.execute(query, *args)
            logger.info(
                "Executed query",
                extra={"query": query.split()[0] if query else "", "result": result},
            )
            return result
        finally:
            await conn.close()
    except Exception:
        logger.exception("Database execution failed", extra={"query": query})
        raise


SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", "60000"))


@dataclass
class OCRResult:
    page_index: int
    text: str
    engine: str
    error: str | None = None
    note: str | None = None


_PADDLE_LOCK = threading.Lock()
_PADDLE_MODELS: dict[str, object] = {}
_PADDLE_ERRORS: dict[str, str] = {}

_PADDLE_LANG_ALIASES = {
    "en": "en",
    "english": "en",
    "en-us": "en",
    "en-gb": "en",
    "es": "es",
    "spanish": "es",
    "fr": "fr",
    "french": "fr",
    "de": "german",
    "german": "german",
    "pt": "pt",
    "pt-br": "pt",
    "portuguese": "pt",
    "it": "it",
    "italian": "it",
    "zh": "ch",
    "zh-cn": "ch",
    "zh-tw": "chinese_cht",
    "ch": "ch",
    "chinese": "ch",
    "ja": "japan",
    "japanese": "japan",
    "ko": "korean",
    "korean": "korean",
    "ru": "ru",
    "russian": "ru",
}


def _normalize_lang_hints(lang_hints: Iterable[str] | str | None) -> list[str]:
    if not lang_hints:
        return []
    if isinstance(lang_hints, str):
        value = lang_hints.strip()
        return [value] if value else []
    hints: list[str] = []
    for hint in lang_hints:
        if hint is None:
            continue
        text = str(hint).strip()
        if text:
            hints.append(text)
    return hints


def _env_flag(name: str, default: str = "0") -> bool:
    val = os.getenv(name, default)
    if val is None:
        return False
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except Exception:
        return default


def _normalize_lang_code(value: str | None) -> str:
    if not value:
        return ""
    raw = value.strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if lowered in _PADDLE_LANG_ALIASES:
        return _PADDLE_LANG_ALIASES[lowered]
    if "-" in lowered:
        prefix = lowered.split("-", 1)[0]
        if prefix in _PADDLE_LANG_ALIASES:
            return _PADDLE_LANG_ALIASES[prefix]
    return raw


def _resolve_paddle_lang(lang_hints: Iterable[str] | str | None) -> str:
    env_lang = _normalize_lang_code(os.getenv("PADDLEOCR_LANG"))
    if env_lang:
        return env_lang
    for hint in _normalize_lang_hints(lang_hints):
        candidate = _normalize_lang_code(hint)
        if candidate:
            return candidate
    return "en"


def _paddle_min_confidence() -> float:
    return _env_float("PADDLEOCR_MIN_CONFIDENCE", 0.5)


def _ensure_paddle_model(lang: str) -> tuple[object | None, str | None]:
    key = (lang or "en").strip().lower() or "en"
    if key in _PADDLE_ERRORS:
        return None, _PADDLE_ERRORS[key]
    model = _PADDLE_MODELS.get(key)
    if model is not None:
        return model, None
    with _PADDLE_LOCK:
        if key in _PADDLE_ERRORS:
            return None, _PADDLE_ERRORS[key]
        model = _PADDLE_MODELS.get(key)
        if model is not None:
            return model, None
        try:
            from paddleocr import PaddleOCR

            kwargs = {
                "lang": key,
                "use_angle_cls": _env_flag("PADDLEOCR_USE_ANGLE_CLS", "1"),
                "use_gpu": _env_flag("PADDLEOCR_USE_GPU", "0"),
            }
            det_dir = os.getenv("PADDLEOCR_DET_MODEL_DIR")
            if det_dir:
                kwargs["det_model_dir"] = det_dir
            rec_dir = os.getenv("PADDLEOCR_REC_MODEL_DIR")
            if rec_dir:
                kwargs["rec_model_dir"] = rec_dir
            cls_dir = os.getenv("PADDLEOCR_CLS_MODEL_DIR")
            if cls_dir:
                kwargs["cls_model_dir"] = cls_dir
            model = PaddleOCR(**kwargs)
            _PADDLE_MODELS[key] = model
            return model, None
        except Exception as exc:  # pragma: no cover - depends on runtime env
            message = f"PaddleOCR init failed ({key}): {exc}"
            _PADDLE_ERRORS[key] = message
            return None, message


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray | None:
    if not image_bytes:
        return None
    try:
        array = np.frombuffer(image_bytes, np.uint8)
        if array.size == 0:
            return None
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


def _iter_paddle_entries(raw) -> Iterable[tuple[str, float | None]]:
    if raw is None:
        return

    def _is_line(entry) -> bool:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            return False
        text_part = entry[1]
        if not isinstance(text_part, (list, tuple)) or not text_part:
            return False
        text = text_part[0]
        return isinstance(text, str)

    stack = [raw]
    while stack:
        current = stack.pop()
        if _is_line(current):
            text_part = current[1]
            text = text_part[0]
            score = None
            if len(text_part) > 1:
                try:
                    score = float(text_part[1])
                except Exception:
                    score = None
            yield text, score
            continue
        if isinstance(current, (list, tuple)):
            for item in current:
                stack.append(item)


def _paddle_result_to_text(raw, min_score: float) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for text, score in _iter_paddle_entries(raw) or []:
        if not text:
            continue
        if score is not None and score < min_score:
            continue
        cleaned = text.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        lines.append(cleaned)
    return "\n".join(lines).strip()


def _ocr_image_with_model(image_bytes: bytes, *, page_index: int, lang_hints: Iterable[str] | str | None) -> OCRResult:
    lang = _resolve_paddle_lang(lang_hints)
    model, init_error = _ensure_paddle_model(lang)
    if model is None:
        return OCRResult(page_index, "", "paddleocr_error", init_error)

    image = _decode_image_bytes(image_bytes)
    if image is None:
        return OCRResult(page_index, "", "paddleocr_error", "decode_failed")

    try:
        raw = model.ocr(image, cls=True)
    except Exception as exc:  # pragma: no cover - runtime specific
        return OCRResult(page_index, "", "paddleocr_error", f"runtime_error: {exc}")

    text = _paddle_result_to_text(raw, _paddle_min_confidence())
    engine = "paddleocr" if text else "paddleocr_empty"
    return OCRResult(page_index, text, engine)


def _iter_pdf_page_images(pdf_bytes: bytes, dpi: float) -> list[tuple[int, bytes]]:
    if not pdf_bytes:
        return []
    try:
        document = pdfium.PdfDocument(pdf_bytes)
    except Exception:
        return []

    scale = max(dpi, 72.0) / 72.0
    page_images: list[tuple[int, bytes]] = []
    try:
        for idx in range(len(document)):
            page = document[idx]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            page_images.append((idx + 1, buffer.getvalue()))
            bitmap.close()
            page.close()
    finally:
        document.close()
    return page_images


def _upsert_page_row(document_id: str, page_index: int, ocr_key: str, status: str, engine: str) -> None:
    try:
        updated = asyncio.run(
            _db_exec(
                "update pages set text_key=$1, status=$2, engine=$3 where document_id=$4 and page_index=$5",
                ocr_key,
                status,
                engine,
                uuid.UUID(document_id),
                page_index,
            )
        )
        needs_insert = True
        if isinstance(updated, str):
            parts = updated.strip().split()
            if len(parts) == 2 and parts[0] == "UPDATE" and parts[1] != "0":
                needs_insert = False
        if needs_insert:
            asyncio.run(
                _db_exec(
                    "insert into pages(id, document_id, page_index, text_key, status, engine) values($1,$2,$3,$4,$5,$6)",
                    uuid.uuid4(),
                    uuid.UUID(document_id),
                    page_index,
                    ocr_key,
                    status,
                    engine,
                )
            )
    except Exception:
        pass


def _store_page_payload(
    client,
    bucket: str,
    document_id: str,
    storage_key: str,
    result: OCRResult,
    *,
    content_type: str | None = None,
    lang_hints: Iterable[str] | str | None = None,
) -> tuple[str, str]:
    ocr_key = f"ocr/{document_id}/page-{result.page_index}.json"
    payload = {
        "document_id": document_id,
        "page_index": result.page_index,
        "storage_key": storage_key,
        "text": result.text,
        "engine": result.engine,
    }
    if content_type:
        payload["content_type"] = content_type
    hints = _normalize_lang_hints(lang_hints)
    if hints:
        payload["lang_hints"] = hints
    if result.error:
        payload["error"] = result.error
    if result.note:
        payload["note"] = result.note
    logger.info(
        "Uploading OCR page",
        extra={
            "document_id": document_id,
            "page_index": result.page_index,
            "status": "pending",
            "ocr_key": ocr_key,
        },
    )
    try:
        client.put_object(
            Bucket=bucket,
            Key=ocr_key,
            Body=json.dumps(payload).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as exc:
        logger.exception(
            "Failed to upload OCR page",
            extra={
                "document_id": document_id,
                "page_index": result.page_index,
                "ocr_key": ocr_key,
            },
        )
        raise

    status = "ocr_done" if result.error is None else "failed"
    _upsert_page_row(document_id, result.page_index, ocr_key, status, result.engine)
    logger.info(
        "Stored OCR page",
        extra={
            "document_id": document_id,
            "page_index": result.page_index,
            "status": status,
            "ocr_key": ocr_key,
            "error": result.error,
        },
    )
    return ocr_key, status


def _extract_page_index(key: str) -> int | None:
    if not key:
        return None
    match = re.search(r"page-(\d+)", key)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    return None


def _summarize_sources(keys: List[str]) -> Tuple[List[Tuple[int | None, str]], str]:
    hints: List[Tuple[int | None, str]] = []
    seen = set()
    for key in keys or []:
        if key in seen:
            continue
        seen.add(key)
        hints.append((_extract_page_index(key), key))
    hints.sort(key=lambda item: (item[0] if item[0] is not None else 10_000, item[1]))
    if not hints:
        return hints, "(no source keys)"
    lines = []
    for page, key in hints:
        if page is not None:
            lines.append(f"- Page {page}: {key}")
        else:
            lines.append(f"- {key}")
    return hints, "\n".join(lines)


def _trim_text_for_model(text: str, max_chars: int = SUMMARY_MAX_CHARS) -> Tuple[str, bool]:
    trimmed = text.strip()
    if len(trimmed) <= max_chars:
        return trimmed, False
    candidate = trimmed[:max_chars]
    # Try to cut on a paragraph boundary for cleaner truncation
    boundary = candidate.rfind("\n\n")
    if boundary < max_chars * 0.6:
        boundary = candidate.rfind("\n")
    if boundary > max_chars * 0.3:
        candidate = candidate[:boundary]
    return candidate.strip(), True


@celery_app.task(name="tasks.echo")
def echo(value):
    """Simple echo task for smoke testing the worker."""
    return value


@celery_app.task(name="tasks.ocr_document")
def ocr_document(
    document_id: str,
    storage_key: str | None = None,
    lang_hints: list[str] | None = None,
    **_,
):
    """OCR a single image-like object using PaddleOCR and store the result."""

    client = s3_client()
    bucket = os.getenv("S3_BUCKET")
    if not bucket or not storage_key:
        return {"error": "missing bucket or storage_key", "document_id": document_id}

    try:
        head = client.head_object(Bucket=bucket, Key=storage_key)
        content_type = (head.get("ContentType") or "").lower()
    except Exception:
        content_type = ""

    # Guard against PDFs routed to the wrong task
    lower_key = (storage_key or "").lower()
    if "pdf" in (content_type or "") or lower_key.endswith(".pdf"):
        skip_result = OCRResult(
            page_index=1,
            text="",
            engine="paddleocr_skip",
            error="pdf_detected",
            note="PDF detected; enqueue tasks.ocr_pdf_document instead.",
        )
        ocr_key, status = _store_page_payload(
            client,
            bucket,
            document_id,
            storage_key,
            skip_result,
            content_type=content_type,
            lang_hints=lang_hints,
        )
        try:
            _run_sync(_db_exec("update documents set status=$1 where id=$2", status, uuid.UUID(document_id)))
        except Exception as exc:
            logger.warning(
                "Failed to update document status (pdf skip): %s",
                exc,
                extra={"document_id": document_id, "status": status},
            )
        return {"document_id": document_id, "ocr_key": ocr_key, "engine": skip_result.engine, "status": status}

    try:
        obj = client.get_object(Bucket=bucket, Key=storage_key)
        body_bytes = obj["Body"].read()
    except Exception as exc:
        ocr_key = f"ocr/{document_id}/page-1.json"
        payload = {
            "document_id": document_id,
            "storage_key": storage_key,
            "content_type": content_type,
            "engine": "paddleocr_error",
            "text": "",
            "error": f"download_failed: {exc}",
        }
        client.put_object(Bucket=bucket, Key=ocr_key, Body=json.dumps(payload).encode("utf-8"), ContentType="application/json")
        try:
            _run_sync(_db_exec("update documents set status=$1 where id=$2", "failed", uuid.UUID(document_id)))
        except Exception as exc:
            logger.warning(
                "Failed to mark document download error: %s",
                exc,
                extra={"document_id": document_id},
            )
        return {"document_id": document_id, "ocr_key": ocr_key, "status": "failed"}

    result = _ocr_image_with_model(body_bytes, page_index=1, lang_hints=lang_hints)
    ocr_key, status = _store_page_payload(
        client,
        bucket,
        document_id,
        storage_key,
        result,
        content_type=content_type,
        lang_hints=lang_hints,
    )
    try:
        _run_sync(_db_exec("update documents set status=$1 where id=$2", status, uuid.UUID(document_id)))
    except Exception as exc:
        logger.warning(
            "Failed to update document status: %s",
            exc,
            extra={"document_id": document_id, "status": status},
        )

    return {"document_id": document_id, "ocr_key": ocr_key, "engine": result.engine, "status": status}


def _openai_client():
    try:
        from openai import OpenAI
    except Exception:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

@celery_app.task(name="tasks.summarize_document")
def summarize_document(document_id: str, text_keys: list[str] | None = None, **_):
    """Summarize OCR text using OpenAI and store notes JSON to assets/.

    Reads ocr/<doc>/page-1.json, extracts text, calls model, and writes
    assets/<doc>/notes.json. Inserts an outputs row in DB if available.
    """
    client = s3_client()
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return {"error": "missing bucket", "document_id": document_id}
    # Determine which OCR text object(s) to read
    text, keys_to_read = _aggregate_document_text(client, bucket, document_id, text_keys)
    source_hints, source_overview = _summarize_sources(keys_to_read)
    summary = None
    engine = "placeholder"
    if text:
        oai = _openai_client()
        if oai is not None:
            try:
                trimmed_text, truncated = _trim_text_for_model(text)
                intro_lines = [
                    f"Document ID: {document_id}",
                    f"Total OCR pages captured: {len(source_hints)}",
                ]
                if truncated:
                    intro_lines.append(
                        f"Note: OCR text truncated to {len(trimmed_text)} of {len(text)} characters for model input."
                    )
                intro_lines.append("Source keys (ordered):")
                intro_lines.append(source_overview)
                intro_lines.append("")
                intro_lines.append("Combined OCR text:")
                user_payload = "\n".join(intro_lines) + "\n" + trimmed_text
                prompt = (
                    "### Role\n"
                    "- You generate study-grade summaries from OCR/parsed JSON for one or more documents.\n"
                    "- You must stay strictly inside the provided content; never add outside facts.\n"
                    "\n### Inputs\n"
                    "- Combined JSON containing text blocks, headings, page numbers (when available), figure/table captions, and metadata per node.\n"
                    "- Treat identical or near-duplicate blocks as duplicates and pick the clearest instance.\n"
                    "\n### Global Rules\n"
                    "- Summaries must remain in the source language unless a target_language control is provided.\n"
                    "- Define jargon briefly the first time it appears.\n"
                    "- Prefer mechanisms, cause→effect, contrasts, and conditional statements.\n"
                    "- If content is missing or unreadable, state that explicitly and avoid unsupported claims.\n"
                    "- Use low creativity (deterministic tone) and consistent terminology.\n"
                    "\n### Output Order (Markdown headings required)\n"
                    "1. Coverage\n"
                    "   - List pages/sections actually used.\n"
                    "   - Note gaps/issues such as missing pages, low OCR confidence, unreadable figures, or duplicates.\n"
                    "2. TL;DR\n"
                    "   - One sentence ≤40 words capturing the main thesis.\n"
                    "   - Exactly three bullets with the most exam-relevant takeaways (facts, mechanisms, or claims).\n"
                    "3. Executive Summary\n"
                    "   - 2–5 short paragraphs (each ~80–120 words).\n"
                    "   - Emphasize key concepts, mechanisms/relationships, implications, and constraints/assumptions.\n"
                    "4. Key Points\n"
                    "   - 8–15 items.\n"
                    "   - For each item provide: point (≤30 words), why it matters (≤25 words), importance score 1–5, tags (comma-separated: definition/process/evidence/etc.), and citations.\n"
                    "5. Outline\n"
                    "   - Ordered list of headings following document order.\n"
                    "   - For each heading or logical section supply page range (or null + location hint) and a 1–3 sentence summary.\n"
                    "\n### Citations (non-negotiable)\n"
                    "- Every factual sentence in TL;DR, Executive Summary, and Key Points must include ≥1 citation.\n"
                    "- Citation format: page number (or null) plus ≤30-word supporting quote and, if page is null, a concrete location hint (e.g., \"Figure 2 caption, right column\").\n"
                    "- Cite only pages/segments listed in Coverage. If support is missing, state \"insufficient evidence\".\n"
                    "\n### Multiple Documents\n"
                    "- Merge overlapping content, remove duplicates, and note disagreements succinctly in the Executive Summary with citations for each viewpoint.\n"
                    "\n### Handling Messy OCR\n"
                    "- Flag unreadable regions in Coverage and avoid depending on them.\n"
                    "- Preserve mathematical symbols exactly and describe relationships in words with citations to the relevant lines.\n"
                    "\n### Compression Profiles\n"
                    "- Lite: TL;DR + 2 paragraphs + 8 Key Points + outline of top-level headings only.\n"
                    "- Standard: TL;DR + 3–4 paragraphs + 12 Key Points + full outline.\n"
                    "- Deep: TL;DR + 5 paragraphs + 15 Key Points + full outline including subheadings.\n"
                    "- Apply the requested profile if a control is provided; otherwise default to Standard.\n"
                    "\n### Failure & Uncertainty\n"
                    "- When evidence is missing, label statements as \"insufficient evidence\" and point to the closest relevant section.\n"
                    "- Do not hedge; clearly distinguish known vs unknown.\n"
                    "\n### Output Skeleton (Markdown only)\n"
                    "`````markdown\n"
                    "# Coverage\n"
                    "- ...\n"
                    "\n# TL;DR\n"
                    "Sentence.\n"
                    "- Bullet\n"
                    "- Bullet\n"
                    "- Bullet\n"
                    "\n# Executive Summary\n"
                    "Paragraph 1\n"
                    "Paragraph 2\n"
                    "...\n"
                    "\n# Key Points\n"
                    "- **Point:** ... | **Why it matters:** ... | **Score:** X | **Tags:** tag1, tag2 | **Citations:** [pX \"quote\"]\n"
                    "...\n"
                    "\n# Outline\n"
                    "1. Heading — pages X–Y — summary...\n"
                    "`````\n"
                    "Return only Markdown following this structure."
                )
                resp = oai.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_payload},
                    ],
                    temperature=0.2,
                )
                summary = resp.choices[0].message.content
                engine = "openai"
            except Exception:
                summary = None
                engine = "openai_error"

    notes = {
        "document_id": document_id,
        "engine": engine,
        "summary": summary or "No OCR text or model unavailable.",
        "sources": keys_to_read,
        "source_hints": [{"page": page, "key": key} for page, key in source_hints],
    }
    notes_key = f"assets/{document_id}/notes.json"
    client.put_object(Bucket=bucket, Key=notes_key, Body=json.dumps(notes).encode("utf-8"), ContentType="application/json")

    # Insert outputs row if DB configured
    try:
        asyncio.run(
            _db_exec(
                "insert into outputs(id, document_id, kind, storage_key) values($1,$2,$3,$4)",
                uuid.uuid4(),
                uuid.UUID(document_id),
                "notes",
                notes_key,
            )
        )
    except Exception:
        pass

    return {"document_id": document_id, "notes_key": notes_key, "engine": engine}


@celery_app.task(name="tasks.summary_generate")
def summary_generate(document_id: str, text_keys: list[str] | None = None):
    client = s3_client()
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return {"error": "missing bucket", "document_id": document_id}
    text, sources = _aggregate_document_text(client, bucket, document_id, text_keys)
    source_hints, source_overview = _summarize_sources(sources)
    out = None
    engine = "placeholder"
    if text:
        oai = _openai_client()
        if oai is not None:
            try:
                trimmed_text, truncated = _trim_text_for_model(text)
                intro_lines = [
                    f"Document ID: {document_id}",
                    f"Total OCR pages captured: {len(source_hints)}",
                ]
                if truncated:
                    intro_lines.append(
                        f"Note: OCR text truncated to {len(trimmed_text)} of {len(text)} characters for model input."
                    )
                intro_lines.append("Source keys (ordered):")
                intro_lines.append(source_overview)
                intro_lines.append("")
                intro_lines.append("Combined OCR text:")
                user_payload = "\n".join(intro_lines) + "\n" + trimmed_text
                prompt = (
                    "You are a careful study assistant. Read the combined OCR text and produce a concise, well-structured study sheet in Markdown only.\n\n"
                    "Requirements:\n"
                    "- Output strictly Markdown, no front-matter or code fences, no explanations about what you are doing.\n"
                    "- Title the sheet 'Key Concepts'.\n"
                    "- Include these sections (omit a line if not inferable from context):\n"
                    "  ## Overview\n"
                    "  - Title: <inferred document title, if any>\n"
                    "  - Author: <inferred author(s), if present>\n"
                    "  - Theme: <one-sentence theme or big idea>\n"
                    "  ## Major Principles (or Main Ideas)\n"
                    "  1. **Term/Concept**: short explanation\n"
                    "  2. **Term/Concept**: short explanation\n"
                    "  3. … (aim for 5–9 items)\n"
                    "  ## Quick Facts / Formulas\n"
                    "  - Fact or formula with a brief hint\n"
                    "  - …\n"
                    "  ## Examples / Applications\n"
                    "  - Example with a short explanation\n"
                    "  - …\n"
                    "  ## Key Terms\n"
                    "  - **Term** — short definition (5–10 terms)\n\n"
                    "Style:\n"
                    "- Use clear headings, numbered/bulleted lists, and **bold** key terms.\n"
                    "- Keep to roughly 250–450 words.\n"
                    "- Avoid redundancy.\n"
                )
                resp = oai.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_payload}],
                    temperature=0.3,
                )
                out = resp.choices[0].message.content
                engine = "openai"
            except Exception:
                out = None
                engine = "openai_error"
    payload = {
        "document_id": document_id,
        "engine": engine,
        "summary": out or "No OCR text or model unavailable.",
        "sources": sources,
        "source_hints": [{"page": page, "key": key} for page, key in source_hints],
    }
    key = f"assets/{document_id}/summary.json"
    client.put_object(Bucket=bucket, Key=key, Body=json.dumps(payload).encode("utf-8"), ContentType="application/json")
    try:
        _run_sync(_db_exec("insert into outputs(id, document_id, kind, storage_key) values($1,$2,$3,$4)", uuid.uuid4(), uuid.UUID(document_id), "summary", key))
    except Exception:
        pass
    return {"document_id": document_id, "summary_key": key, "engine": engine}


def _aggregate_document_text(client, bucket: str, document_id: str, text_keys: list[str] | None = None):
    texts: list[str] = []
    keys_to_read: list[str] = []
    if text_keys:
        keys_to_read = text_keys
    else:
        # Try to get all page text keys from DB; fallback to page-1
        try:
            import asyncpg  # type: ignore
            async def _fetch_keys():
                dsn = _canonicalize_db_url(os.getenv("DATABASE_URL", ""))
                if not dsn:
                    return []
                conn = await asyncpg.connect(dsn)
                try:
                    rows = await conn.fetch(
                        "select text_key from pages where document_id=$1 and text_key is not null order by page_index",
                        uuid.UUID(document_id),
                    )
                    return [r["text_key"] for r in rows]
                finally:
                    await conn.close()
            keys_to_read = asyncio.run(_fetch_keys()) or []
        except Exception:
            keys_to_read = []
        if not keys_to_read:
            keys_to_read = [f"ocr/{document_id}/page-1.json"]

    for k in keys_to_read:
        try:
            obj = client.get_object(Bucket=bucket, Key=k)
            payload = json.loads(obj["Body"].read().decode("utf-8"))
            t = (payload.get("text") or "").strip()
            if t:
                texts.append(t)
        except Exception:
            continue
    return ("\n\n".join(texts).strip(), keys_to_read)


def _split_paragraphs(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def _collect_subject_windows(text: str, subjects: list[str] | None) -> dict[str, list[str]]:
    if not text or not subjects:
        return {}
    paragraphs = _split_paragraphs(text)
    lowered = [p.lower() for p in paragraphs]
    windows: dict[str, list[str]] = {}
    for raw in subjects:
        name = (raw or "").strip()
        if not name:
            continue
        key = name.lower()
        highlights: list[str] = []
        tokens = [tok for tok in re.findall(r"[\w-]{4,}", key) if tok]
        if not tokens:
            tokens = [key]
        for idx, para in enumerate(paragraphs):
            para_lower = lowered[idx]
            if key in para_lower or any(tok in para_lower for tok in tokens):
                candidates = []
                if idx > 0:
                    candidates.append(paragraphs[idx - 1])
                candidates.append(paragraphs[idx])
                if idx + 1 < len(paragraphs):
                    candidates.append(paragraphs[idx + 1])
                for snippet in candidates:
                    snippet_clean = snippet.strip()
                    if not snippet_clean:
                        continue
                    trimmed = snippet_clean[:400].rstrip()
                    if len(snippet_clean) > 400:
                        trimmed = trimmed + "…"
                    if trimmed not in highlights:
                        highlights.append(trimmed)
                if len(highlights) >= 5:
                    break
        if highlights:
            windows[name] = highlights[:5]
    return windows


def _combine_subject_text(text: str, subjects: list[str] | None) -> tuple[str, dict[str, list[str]]]:
    subject_windows = _collect_subject_windows(text, subjects)
    if not subject_windows:
        return text, {}
    ordered: list[str] = []
    seen = set()
    for topic in subjects or []:
        highlights = subject_windows.get(topic, [])
        for snippet in highlights:
            if snippet in seen:
                continue
            seen.add(snippet)
            ordered.append(snippet)
    combined = "\n\n".join(ordered)
    if not combined.strip():
        return text, subject_windows
    return combined.strip(), subject_windows


def _estimate_question_count(text: str, subjects: list[str] | None, target: int | None = None) -> int:
    if target:
        try:
            return max(1, min(50, int(target)))
        except Exception:
            pass
    word_count = len(re.findall(r"\w+", text))
    if word_count <= 0:
        return 0
    base = 5
    thresholds = [
        (220, 6),
        (320, 8),
        (480, 10),
        (650, 12),
        (900, 14),
        (1200, 16),
        (1600, 20),
        (2200, 24),
    ]
    for limit, value in thresholds:
        if word_count > limit:
            base = value
    auto_cap = max(3, min(30, word_count // 35 + 4))
    base = min(base, auto_cap)
    if subjects:
        per_topic = 3 if word_count >= 700 else 2
        if word_count >= 1500:
            per_topic = 4
        base = max(base, min(30, per_topic * len(subjects)))
    return min(30, max(3, base))


@celery_app.task(name="tasks.ocr_page")
def ocr_page(
    document_id: str | None = None,
    page_index: int | None = None,
    storage_key: str | None = None,
    lang_hints=None,
    # Backward-compat arg names (old producer)
    page_id: str | None = None,
    s3_key_raw: str | None = None,
    **_,
):
    """OCR a single page and update DB + S3.

    - Reads raw object `storage_key`
    - Writes text JSON to `ocr/<document_id>/page-<page_index>.json`
    - Updates pages row status and text_key
    - If all pages for the doc are done, updates document status to ocr_done
    """
    # Coalesce legacy/new arg shapes
    storage_key = storage_key or s3_key_raw
    try:
        page_index = int(page_index) if page_index is not None else (int(page_id) if page_id is not None else None)
    except Exception:
        page_index = page_index or 1

    client = s3_client()
    bucket = os.getenv("S3_BUCKET")
    if not bucket or not document_id or not storage_key:
        return {"error": "missing bucket/document_id/storage_key", "document_id": document_id, "page_index": page_index}

    # Download object bytes
    try:
        obj = client.get_object(Bucket=bucket, Key=storage_key)
        body_bytes = obj["Body"].read()
        content_type = obj.get("ContentType", "")
    except Exception as e:
        return {"error": f"download_failed: {e}", "document_id": document_id, "page_index": page_index}

    ocr_result = _ocr_image_with_model(body_bytes, page_index=page_index or 1, lang_hints=lang_hints)

    ocr_key = f"ocr/{document_id}/page-{page_index or 1}.json"
    payload = {
        "document_id": document_id,
        "page_index": page_index,
        "storage_key": storage_key,
        "content_type": content_type,
        "text": ocr_result.text,
        "engine": ocr_result.engine,
    }
    hints = _normalize_lang_hints(lang_hints)
    if hints:
        payload["lang_hints"] = hints
    if ocr_result.error:
        payload["error"] = ocr_result.error
    client.put_object(Bucket=bucket, Key=ocr_key, Body=json.dumps(payload).encode("utf-8"), ContentType="application/json")

    # Update page row
    try:
        asyncio.run(
            _db_exec(
                "update pages set text_key=$1, status=$2, engine=$3 where document_id=$4 and page_index=$5",
                ocr_key,
                "ocr_done" if ocr_result.error is None else "failed",
                ocr_result.engine,
                uuid.UUID(document_id),
                page_index,
            )
        )
        # If all pages done, update document
        import asyncpg  # type: ignore
        async def _check_all_done():
            dsn = _canonicalize_db_url(os.getenv("DATABASE_URL", ""))
            if not dsn:
                return False
            conn = await asyncpg.connect(dsn)
            try:
                total = await conn.fetchval("select count(*) from pages where document_id=$1", uuid.UUID(document_id))
                done = await conn.fetchval("select count(*) from pages where document_id=$1 and status='ocr_done'", uuid.UUID(document_id))
                if total and done == total:
                    row = await conn.fetchrow("select status from documents where id=$1", uuid.UUID(document_id))
                    status = row["status"] if row else None
                    if status != "ocr_done":
                        await conn.execute("update documents set status='ocr_done' where id=$1", uuid.UUID(document_id))
                        return True
            finally:
                await conn.close()
            return False
        done_all = asyncio.run(_check_all_done())
    except Exception:
        done_all = False
    if done_all:
        try:
            celery_app.send_task("tasks.flashcards_generate", kwargs={"document_id": document_id, "count": 20})
        except Exception:
            pass
    return {"document_id": document_id, "page_index": page_index, "ocr_key": ocr_key}


@celery_app.task(name="tasks.ocr_pdf_document")
def ocr_pdf_document(
    document_id: str,
    storage_key: str,
    lang_hints: list[str] | None = None,
    **_,
):
    """OCR a PDF by rendering pages locally and running PaddleOCR per page."""

    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return {"error": "missing B2 bucket", "document_id": document_id}

    logger.info(
        "Starting PDF OCR",
        extra={
            "document_id": document_id,
            "storage_key": storage_key,
            "bucket": bucket,
        },
    )

    client = s3_client()
    try:
        pdf_obj = client.get_object(Bucket=bucket, Key=storage_key)
        pdf_bytes = pdf_obj["Body"].read()
        content_type = (pdf_obj.get("ContentType") or "application/pdf").lower()
    except Exception as exc:
        logger.exception(
            "Failed to download PDF",
            extra={"document_id": document_id, "storage_key": storage_key},
        )
        return {"error": f"download_pdf_failed: {exc}", "document_id": document_id}

    pdf_dpi = _env_float("PADDLEOCR_PDF_DPI", 180.0)
    try:
        image_pages = _iter_pdf_page_images(pdf_bytes, pdf_dpi)
        render_error = None
    except Exception as exc:
        image_pages = []
        render_error = f"pdf_render_failed: {exc}"
        logger.exception(
            "Failed to render PDF pages",
            extra={"document_id": document_id, "storage_key": storage_key},
        )

    if not image_pages:
        failure = OCRResult(page_index=1, text="", engine="paddleocr_error", error=render_error or "pdf_empty")
        ocr_key, status = _store_page_payload(
            client,
            bucket,
            document_id,
            storage_key,
            failure,
            content_type=content_type,
            lang_hints=lang_hints,
        )
        try:
            _run_sync(_db_exec("update documents set status=$1, page_count=$2 where id=$3", status, 0, uuid.UUID(document_id)))
        except Exception as exc:
            logger.warning(
                "Failed to update document status for empty PDF: %s",
                exc,
                extra={"document_id": document_id, "status": status},
            )
        logger.warning(
            "PDF produced no pages",
            extra={
                "document_id": document_id,
                "storage_key": storage_key,
                "status": status,
                "error": failure.error,
            },
        )
        return {"document_id": document_id, "pages": 0, "engine": failure.engine, "status": status}

    statuses: list[str] = []
    for page_index, image_bytes in image_pages:
        result = _ocr_image_with_model(image_bytes, page_index=page_index, lang_hints=lang_hints)
        page_result = OCRResult(page_index, result.text or "", result.engine, result.error)
        _, status = _store_page_payload(
            client,
            bucket,
            document_id,
            storage_key,
            page_result,
            content_type=content_type,
            lang_hints=lang_hints,
        )
        statuses.append(status)
        logger.info(
            "Processed PDF page",
            extra={
                "document_id": document_id,
                "page_index": page_index,
                "status": status,
                "engine": page_result.engine,
                "error": page_result.error,
            },
        )

    page_count = len(image_pages)
    doc_status = "ocr_done" if statuses and all(s == "ocr_done" for s in statuses) else "failed"
    try:
        _run_sync(
            _db_exec(
                "update documents set status=$1, page_count=$2 where id=$3",
                doc_status,
                page_count,
                uuid.UUID(document_id),
            )
        )
    except Exception as exc:
        logger.warning(
            "Failed to update document aggregate status: %s",
            exc,
            extra={"document_id": document_id, "status": doc_status},
        )

    logger.info(
        "Completed PDF OCR",
        extra={
            "document_id": document_id,
            "storage_key": storage_key,
            "pages": page_count,
            "status": doc_status,
        },
    )

    if doc_status == "ocr_done":
        try:
            celery_app.send_task("tasks.flashcards_generate", kwargs={"document_id": document_id, "count": 20})
        except Exception:
            pass

    return {"document_id": document_id, "pages": page_count, "engine": "paddleocr", "status": doc_status}


@celery_app.task(name="tasks.flashcards_generate")
def flashcards_generate(document_id: str, count: int = 20):
    client = s3_client()
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return {"error": "missing bucket", "document_id": document_id}
    text, sources = _aggregate_document_text(client, bucket, document_id, None)
    cards = []
    engine = "placeholder"
    try:
        desired_count = max(int(count), 0)
    except Exception:
        desired_count = 0
    if text:
        oai = _openai_client()
        if oai is not None:
            try:
                trimmed_text, truncated = _trim_text_for_model(text)
                word_count = len(trimmed_text.split())
                meta_lines = [
                    f"Document ID: {document_id}",
                    f"Approximate word count in excerpt: {word_count}",
                    "Enumerate all distinct, high-value flashcards you can derive from this content.",
                    "Do not limit yourself to an arbitrary target count.",
                    "Skip any concept that is redundant, trivial, or insufficiently supported.",
                ]
                if truncated:
                    meta_lines.append("Note: OCR text truncated to fit model limits; focus on covered sections only.")
                if desired_count:
                    meta_lines.append(
                        f"Aim to cover at least {desired_count} distinct flashcards when the document supports that depth."
                    )
                meta_lines.append("")
                meta_lines.append("Combined OCR text:")
                user_payload = "\n".join(meta_lines) + "\n" + trimmed_text
                target_requirement = (
                    f"- Aim to deliver at least {desired_count} distinct flashcards when the source material supports it.\n"
                    if desired_count
                    else ""
                )
                prompt = (
                    "### Role\n"
                    "- You are an expert learning designer who creates rigorous, exam-ready flashcards using only the supplied document content.\n"
                    "- You must stay within the provided material—no outside facts, paraphrasing only what is supported.\n"
                    "\n### Input\n"
                    "- OCR/parsed JSON containing text blocks, headings, page numbers (when available), captions, and metadata.\n"
                    "- Some blocks may repeat; consolidate duplicates and rely on the clearest instance.\n"
                    "\n### Output Requirements\n"
                    "- Return a JSON array with one object per worthwhile flashcard. Do not force an upper bound.\n"
                    f"{target_requirement}"
                    "- Create a card only when the underlying fact/mechanism is clearly supported and genuinely helpful for spaced repetition.\n"
                    "- Think step-by-step: scan all sections, cluster related concepts, rank by importance, then write cards.\n"
                    "\n### Flashcard Object Schema\n"
                    "Each object must contain:\n"
                    "- question: short prompt (≤140 chars) that can stand alone.\n"
                    "- answer: rigorous answer (≤220 chars) that fully resolves the question.\n"
                    "- type: one of definition, process, mechanism, comparison, evidence, application.\n"
                    "- difficulty: integer 1–5 (1=easiest).\n"
                    "- tags: array of 1–4 keywords (e.g., ['enzyme', 'regulation']).\n"
                    "- source: object with keys page (number|null) and quote (≤30 words). If page is null, add location_hint describing where in the text it appears.\n"
                    "\n### Quality Rules\n"
                    "- Ground every question/answer in the source; cite the exact supporting quote.\n"
                    "- Prefer mechanisms, definitions, comparisons, prerequisites, and cause→effect relationships that aid spaced repetition.\n"
                    "- Avoid trivial facts, generic terminology, or duplicative cards.\n"
                    "- Rephrase content into question/answer form; do not copy raw sentences verbatim.\n"
                    "\n### Failure Handling\n"
                    "- If the document truly contains no eligible facts, return an empty JSON array [].\n"
                    "- Do not include commentary, Markdown, or code fences—only the JSON array.\n"
                    "- Maintain deterministic tone (temperature ≈ 0).\n"
                )
                resp = oai.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_payload},
                    ],
                    temperature=0.3,
                )
                raw = resp.choices[0].message.content or "[]"
                try:
                    cards = json.loads(raw)
                except Exception:
                    # Try to extract JSON block if model wrapped
                    import re
                    m = re.search(r"\[.*\]", raw, re.S)
                    if m:
                        cards = json.loads(m.group(0))
                engine = "openai"
            except Exception:
                cards = []
                engine = "openai_error"
    data = {"document_id": document_id, "engine": engine, "flashcards": cards, "sources": sources}
    key = f"assets/{document_id}/flashcards.json"
    client.put_object(Bucket=bucket, Key=key, Body=json.dumps(data).encode("utf-8"), ContentType="application/json")
    try:
        asyncio.run(
            _db_exec(
                "insert into outputs(id, document_id, kind, storage_key) values($1,$2,$3,$4)",
                uuid.uuid4(),
                uuid.UUID(document_id),
                "flashcards",
                key,
            )
        )
    except Exception:
        pass
    return {"document_id": document_id, "flashcards_key": key, "engine": engine}


def _acquire_lock(name: str, ttl: int = 60) -> bool:
    try:
        import redis  # type: ignore
        url = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        r = redis.StrictRedis.from_url(url)
        return bool(r.set(name, "1", nx=True, ex=ttl))
    except Exception:
        return True  # best-effort


def _release_lock(name: str) -> None:
    try:
        import redis  # type: ignore
        url = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        r = redis.StrictRedis.from_url(url)
        r.delete(name)
    except Exception:
        pass


@celery_app.task(name="tasks.quiz_topics")
def quiz_topics(document_id: str, limit: int = 12):
    client = s3_client()
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return {"error": "missing bucket", "document_id": document_id}
    text, _ = _aggregate_document_text(client, bucket, document_id, None)
    topics: list[str] = []
    engine = "openai"
    if text:
        oai = _openai_client()
        if oai is not None:
            try:
                tcount = min(20, max(3, int(limit or 8)))
                prompt = (
                    "Identify the most quiz-worthy subjects in the provided study material. "
                    "Return a strict JSON array of concise topic strings (no explanations). "
                    "Focus on concepts students could be quizzed on. Limit to {N} unique items."
                ).replace("{N}", str(tcount))
                resp = oai.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text[:8000]},
                    ],
                    temperature=0.2,
                )
                raw = resp.choices[0].message.content or "[]"
                parsed: list[str] = []
                try:
                    data = json.loads(raw)
                    if isinstance(data, list):
                        parsed = [str(x).strip() for x in data if str(x).strip()]
                except Exception:
                    match = re.search(r"\[.*\]", raw, re.S)
                    if match:
                        try:
                            data = json.loads(match.group(0))
                            if isinstance(data, list):
                                parsed = [str(x).strip() for x in data if str(x).strip()]
                        except Exception:
                            parsed = []
                deduped: list[str] = []
                seen = set()
                for item in parsed:
                    key = item.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(item)
                    if len(deduped) >= tcount:
                        break
                topics = deduped
            except Exception:
                engine = "openai_error"
                topics = []
    subject_windows = _collect_subject_windows(text, topics)
    data = {
        "document_id": document_id,
        "topics": topics,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "engine": engine,
        "subjects": [
            {"name": topic, "highlights": subject_windows.get(topic, [])}
            for topic in topics
        ],
    }
    key = f"assets/{document_id}/quiz_topics.json"
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data).encode("utf-8"),
        ContentType="application/json",
    )
    return {"document_id": document_id, "topics_key": key, "topics": topics, "engine": engine}


@celery_app.task(name="tasks.quiz_generate")
def quiz_generate(document_id: str, target_count: int | None = None, qtype: str = "mcq", topics: list[str] | None = None):
    client = s3_client()
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return {"error": "missing bucket", "document_id": document_id}
    text, sources = _aggregate_document_text(client, bucket, document_id, None)
    prompt_text, subject_windows = _combine_subject_text(text, topics)
    text_for_prompt = prompt_text or text
    desired_count = _estimate_question_count(text_for_prompt, topics, target_count)
    if desired_count <= 0:
        desired_count = 5 if text_for_prompt else 0
    kind = (qtype or "mcq").lower()
    quiz: list[dict] = []
    engine = "placeholder"
    oai = None
    text_lower = text_for_prompt.lower() if text_for_prompt else ""

    def _build_shape(n: int) -> str:
        if kind == "tf":
            return (
                "Return exactly {N} true/false questions as a JSON array. Each item must be {"
                "question: string, options: [\"True\", \"False\"], answer: 1|2, explanation: string}."
            ).replace("{N}", str(n))
        if kind == "cloze":
            return (
                "Return exactly {N} fill-in-the-blank questions as a JSON array. Each item must be {"
                "question: string containing a ____ blank, options: [correct, distractor1, distractor2, distractor3],"
                " answer: 1, explanation: string}."
            ).replace("{N}", str(n))
        return (
            "Return exactly {N} multiple-choice questions as a JSON array. Each item must be {"
            "question: string, options: [option1, option2, option3, option4], answer: 1..4, explanation: string}."
        ).replace("{N}", str(n))

    def _prompt_header(extra: str = "") -> str:
        topics_hint = ""
        if topics:
            topics_hint = " Focus on these topics only when they are covered in the source: " + ", ".join(topics)
        return (
            "You are creating study questions strictly grounded in the provided document excerpt. "
            "Never invent facts or rely on outside knowledge. Ask about details that appear in the excerpt only."
            + topics_hint
            + extra
        )

    def _normalize(items: list[dict]) -> list[dict]:
        out: list[dict] = []
        for it in items or []:
            q = (it.get("question") or "").strip()
            opts = it.get("options") or []
            ans = it.get("answer")
            exp = (it.get("explanation") or "").strip()
            if not q or not isinstance(opts, list) or len(opts) < 2:
                continue
            if text_lower:
                words = re.findall(r"[a-zA-Z]{4,}", q.lower())
                if words:
                    matches = {w for w in words if w in text_lower}
                    if not matches:
                        # Skip questions that do not reference the source material.
                        continue
                    if len(words) >= 4 and len(matches) < 2:
                        # Require at least two overlapping keywords for longer questions.
                        continue
            if kind == "tf":
                opts = ["True", "False"]
                if isinstance(ans, bool):
                    ans = 1 if ans else 2
                if str(ans) not in {"1", "2", 1, 2}:
                    ans = 1
            try:
                ans_i = int(ans)
            except Exception:
                ans_i = 1
            if ans_i < 1 or ans_i > len(opts):
                ans_i = 1
            cleaned_opts = [str(x).strip() for x in opts[:4] if str(x).strip()]
            if kind != "tf" and len(cleaned_opts) < 4:
                # Backfill missing choices with distinct distractors from the source text if possible.
                fillers: list[str] = []
                if text_lower:
                    snippets = re.findall(r"[A-Z][^\.\n]{15,100}", text_for_prompt)
                    for snippet in snippets:
                        snippet_clean = snippet.strip()
                        if snippet_clean and snippet_clean not in cleaned_opts and len(fillers) < (4 - len(cleaned_opts)):
                            fillers.append(snippet_clean[:60])
                while len(cleaned_opts) + len(fillers) < 4:
                    fillers.append(f"Choice {len(cleaned_opts) + len(fillers) + 1}")
                cleaned_opts.extend(fillers[: max(0, 4 - len(cleaned_opts))])
            out.append(
                {
                    "question": q,
                    "options": cleaned_opts if cleaned_opts else [str(x) for x in opts[:4]],
                    "answer": ans_i,
                    "explanation": exp,
                }
            )
        return out

    if text_for_prompt:
        oai = _openai_client()
        if oai is not None:
            try:
                system_prompt = _prompt_header(" " + _build_shape(desired_count))
                resp = oai.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": system_prompt + " Respond with strict JSON only."},
                        {
                            "role": "user",
                            "content": "Document excerpt:\n" + text_for_prompt[:16000],
                        },
                    ],
                    temperature=0.2,
                )
                raw = resp.choices[0].message.content or "[]"
                try:
                    quiz = json.loads(raw)
                except Exception:
                    match = re.search(r"\[.*\]", raw, re.S)
                    quiz = json.loads(match.group(0)) if match else []
                engine = "openai"
            except Exception:
                quiz = []
                engine = "openai_error"

    normalized = _normalize(quiz)
    lock_key = f"lock:quiz:{document_id}:{kind}:{desired_count or 'auto'}"
    got_lock = _acquire_lock(lock_key, ttl=120)
    try:
        tries = 0
        while text_for_prompt and oai is not None and desired_count and len(normalized) < desired_count and tries < 5:
            tries += 1
            remain = desired_count - len(normalized)
            existing_stems = [entry["question"][:120] for entry in normalized[:10]]
            stem_hint = ""
            if existing_stems:
                stem_hint = " Avoid repeating these question stems: " + " | ".join(existing_stems)
            try:
                system_prompt = _prompt_header(stem_hint + " " + _build_shape(remain))
                resp2 = oai.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": system_prompt + " Respond with strict JSON only."},
                        {
                            "role": "user",
                            "content": "Document excerpt:\n" + text_for_prompt[:16000],
                        },
                    ],
                    temperature=0.2,
                )
                raw2 = resp2.choices[0].message.content or "[]"
                try:
                    additions = json.loads(raw2)
                except Exception:
                    match2 = re.search(r"\[.*\]", raw2, re.S)
                    additions = json.loads(match2.group(0)) if match2 else []
                pool = _normalize(additions)
                if not pool:
                    continue
                seen = {x["question"].strip().lower() for x in normalized}
                for entry in pool:
                    key = entry["question"].strip().lower()
                    if key in seen:
                        continue
                    normalized.append(entry)
                    seen.add(key)
                    if len(normalized) >= desired_count:
                        break
            except Exception:
                continue
        if text_for_prompt and oai is not None and desired_count and len(normalized) < desired_count:
            try:
                system_prompt = _prompt_header(" Regenerate the full set. " + _build_shape(desired_count))
                resp3 = oai.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": system_prompt + " Respond with strict JSON only."},
                        {
                            "role": "user",
                            "content": "Document excerpt:\n" + text_for_prompt[:16000],
                        },
                    ],
                    temperature=0.2,
                )
                raw3 = resp3.choices[0].message.content or "[]"
                try:
                    replacement = json.loads(raw3)
                except Exception:
                    match3 = re.search(r"\[.*\]", raw3, re.S)
                    replacement = json.loads(match3.group(0)) if match3 else []
                refreshed = _normalize(replacement)
                if len(refreshed) >= len(normalized):
                    normalized = refreshed
            except Exception:
                pass
    finally:
        if got_lock:
            _release_lock(lock_key)

    final = normalized[:desired_count] if desired_count else normalized
    data = {
        "document_id": document_id,
        "engine": engine,
        "quiz": final,
        "sources": sources,
        "type": kind,
        "requested_count": len(final),
        "topics": [str(x) for x in (topics or []) if str(x).strip()],
        "count": len(final),
        "subjects": [
            {"name": topic, "highlights": subject_windows.get(topic, [])}
            for topic in (topics or [])
        ],
    }
    key = f"assets/{document_id}/quiz.json"
    client.put_object(Bucket=bucket, Key=key, Body=json.dumps(data).encode("utf-8"), ContentType="application/json")
    try:
        asyncio.run(
            _db_exec(
                "insert into outputs(id, document_id, kind, storage_key) values($1,$2,$3,$4)",
                uuid.uuid4(),
                uuid.UUID(document_id),
                "quiz",
                key,
            )
        )
    except Exception:
        pass
    return {"document_id": document_id, "quiz_key": key, "engine": engine}


@celery_app.task(name="tasks.plan_generate")
def plan_generate(document_id: str, exam_date: str, daily_minutes: int = 60):
    return {"document_id": document_id, "plan_key": f"assets/{document_id}/plan.json", "exam_date": exam_date}


@celery_app.task(name="tasks.env_diag")
def env_diag():
    """Return a simple view of critical env configuration (booleans only)."""
    return {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "gcv_credentials": bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")),
        "gcs_bucket": bool(os.getenv("GCS_BUCKET")),
        "redis_url": bool(os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL")),
        "database_url": bool(os.getenv("DATABASE_URL")),
        "s3": bool(os.getenv("S3_ENDPOINT_URL") and os.getenv("S3_BUCKET") and os.getenv("S3_ACCESS_KEY_ID") and os.getenv("S3_SECRET_ACCESS_KEY")),
    }


def _delete_prefix(client, bucket: str, prefix: str) -> int:
    if not client or not bucket or not prefix:
        return 0
    deleted = 0
    token = None
    while True:
        if token:
            resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, ContinuationToken=token)
        else:
            resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        contents = resp.get("Contents") or []
        if not contents:
            break
        ids = [{"Key": it["Key"]} for it in contents]
        client.delete_objects(Bucket=bucket, Delete={"Objects": ids, "Quiet": True})
        deleted += len(ids)
        if resp.get("IsTruncated") and resp.get("NextContinuationToken"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return deleted


@celery_app.task(name="tasks.storage_cleanup_document")
def storage_cleanup_document(document_id: str, storage_key: str | None = None):
    """Delete S3/B2 objects for a document in the background.

    - Deletes prefixes: assets/<doc>/, ocr/<doc>/, raw/<doc>/ (multi-upload)
    - Deletes the top-level storage_key if provided (single-upload)
    """
    try:
        client = s3_client()
        bucket = os.getenv("S3_BUCKET")
        if not bucket:
            return {"deleted": 0}
        total = 0
        total += _delete_prefix(client, bucket, f"assets/{document_id}/")
        total += _delete_prefix(client, bucket, f"ocr/{document_id}/")
        total += _delete_prefix(client, bucket, f"raw/{document_id}/")
        if storage_key:
            try:
                client.delete_object(Bucket=bucket, Key=storage_key)
                total += 1
            except Exception:
                pass
        return {"deleted": total}
    except Exception:
        return {"deleted": 0}
