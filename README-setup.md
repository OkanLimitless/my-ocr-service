# Worker (Celery)

Celery workers run the document OCR pipeline and LLM post-processing jobs. OCR now relies on [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) so the worker can run fully on self-hosted GPU instances such as RunPod.

## Run Locally

1. Use Python 3.11 (matching the Dockerfile).
2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   - GPU pods: uninstall the CPU wheel and install the GPU build that matches your CUDA version (see [Paddle install guide](https://www.paddlepaddle.org.cn/en/install/guide)). Example for CUDA 12.1:
     ```bash
     pip uninstall -y paddlepaddle
     pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/paddle3.11_gpu/index.html
     ```
3. Start the worker:
   ```bash
   celery -A celery_app.celery_app worker -Q ${CELERY_QUEUE:-default} -l info
   ```
4. Smoke test from a Python REPL:
   ```py
   from tasks import echo
   echo.delay({"hello": "world"})
   ```

## PaddleOCR Configuration

Key environment variables (all optional unless noted):

- `PADDLEOCR_LANG` — language code (`en`, `fr`, `german`, `es`, `ch`, etc.). Defaults to `en`.
- `PADDLEOCR_USE_GPU` — set to `1`/`true` to enable GPU execution.
- `PADDLEOCR_MIN_CONFIDENCE` — filter out detections below this confidence (default `0.5`).
- `PADDLEOCR_PDF_DPI` — rendering DPI for PDF pages (default `180`).
- `PADDLEOCR_DET_MODEL_DIR`, `PADDLEOCR_REC_MODEL_DIR`, `PADDLEOCR_CLS_MODEL_DIR` — override model directories when shipping custom checkpoints.

Language hints passed on tasks (e.g. `lang_hints=["en"]`) are merged with the env default.

## Queue Layout

We split Celery work across two queues:

- `default` — summaries, flashcards, quizzes, diagnostics, etc. (Fly worker stays on this queue).
- `ocr` — OCR-only workloads (`tasks.ocr_document`, `tasks.ocr_page`, `tasks.ocr_pdf_document`). RunPod workers attach here when you need GPU OCR bursts.

Environment variables:

- `CELERY_DEFAULT_QUEUE` (fallbacks to `CELERY_QUEUE` or `default`) — queue name for everything except OCR tasks.
- `CELERY_OCR_QUEUE` (defaults to `ocr`) — queue that OCR tasks are routed to.

Recommended setup:

1. **API**: export both `CELERY_DEFAULT_QUEUE=default` and `CELERY_OCR_QUEUE=ocr` (or your chosen names). This makes task routing explicit.
2. **Fly worker**: keep `CELERY_QUEUE=default` (or `CELERY_DEFAULT_QUEUE=default`) so it only consumes non-OCR jobs.
3. **RunPod worker**: set `CELERY_QUEUE=ocr` (and optionally `CELERY_OCR_QUEUE=ocr`) to consume only OCR jobs. Start it when you need extra capacity; shut it down once the queue drains.

## Required Secrets & Env Vars

| Purpose | Variable |
| --- | --- |
| Redis broker/result backend | `REDIS_URL` **or** `CELERY_BROKER_URL` & `CELERY_RESULT_BACKEND` |
| Task queues | `CELERY_DEFAULT_QUEUE` (fallback: `CELERY_QUEUE` / `default`), `CELERY_OCR_QUEUE` (default `ocr`) |
| Object storage (Backblaze B2 / S3 API) | `S3_ENDPOINT_URL`, `S3_REGION`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_BUCKET` |
| Database writes (optional) | `DATABASE_URL` |
| Downstream LLMs | `OPENAI_API_KEY`, `OPENAI_MODEL` (optional) |

## RunPod Quickstart

1. Provision a GPU pod with Ubuntu 22.04, CUDA ≥ 11.7, and persistent volume.
2. Install system libraries once per pod:
   ```bash
   sudo apt-get update
   sudo apt-get install -y ffmpeg libgomp1 libglib2.0-0 libopenblas-base libsm6 libxext6 libxrender1
   ```
3. Clone the repo or sync the `apps/worker` directory to the pod.
4. Create a venv and install Python deps:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r apps/worker/requirements.txt
   # (Optional) replace with GPU wheel per Paddle docs
   ```
5. Export env vars (see table above). For secrets use RunPod environment variables or a `.env` sourced before starting the worker.
6. Launch Celery (persistent pod):
   ```bash
   CELERY_QUEUE=ocr celery -A celery_app.celery_app worker -l info -c ${CELERY_CONCURRENCY:-4}
   ```

### RunPod Serverless handler

- Entry point: `serverless_handler.handler`
- Expected payload:
  ```json
  {
    "input": {
      "task": "tasks.ocr_document",  // or tasks.ocr_page / tasks.ocr_pdf_document
      "payload": { ... }              // same kwargs we pass to Celery
    }
  }
  ```
- Returns `{ "status": "ok", "result": ... }` on success, `{ "status": "error", "error": "..." }` on failure.
- Configure your RunPod serverless container to call this handler and set the same environment variables needed for storage/Redis/etc.

## Fly.io (legacy deployment)

The existing `Dockerfile` and `fly.toml` still work if you prefer Fly.io. Set Fly secrets to match the table above plus any Paddle-specific overrides.

---
- `tasks.ocr_document` now runs PaddleOCR over images / single-page uploads.
- `tasks.ocr_page` handles per-page jobs with PaddleOCR.
- `tasks.ocr_pdf_document` renders PDFs via pypdfium2 and applies PaddleOCR per page.
