# My OCR Service

Welcome to `my-ocr-service`. This repository currently contains the initial project scaffold and serves as the baseline for future OCR-related development.

## Getting Started

1. Clone the repository: `git clone https://github.com/OkanLimitless/my-ocr-service.git`
2. Navigate into the directory: `cd my-ocr-service`
3. Add your OCR implementation details.

## RunPod Setup

### One-time environment prep

Run the helper script (requires `sudo`) to install OS packages, create the virtualenv, and install Python dependencies:

```bash
sudo ./scripts/runpod_setup.sh
```

### Serverless handler

RunPod Serverless invokes `serverless_handler.handler`. The API submits payloads that mirror the Celery task signature:

```json
{
  "input": {
    "task": "tasks.ocr_document",
    "payload": {
      "document_id": "…",
      "storage_key": "…",
      "lang_hints": ["en"]
    }
  }
}
```

For ad-hoc testing you can still send `url` or `base64` fields—the handler will decode them and call the same task logic.

The container automatically starts the serverless loop when executed as a module (see `runpod.yaml`). Set `RUNPOD_DISABLE_SERVERLESS=1` when running locally to bypass the loop (for unit tests or manual invocations).

PaddleOCR weights are downloaded inside the Docker build, so cold starts do not fetch models. The container now ships on `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`, adds NVIDIA’s CUDA apt repo, installs the latest cuDNN/NCCL runtime and dev packages, and runs `paddle.utils.run_check()` during build after installing the CUDA 12.1 Paddle wheel (`paddlepaddle-gpu==2.6.1.post121`). PaddleOCR runs on GPU by default (`PADDLEOCR_USE_GPU=1`). When switching languages, adjust the download URLs in the `Dockerfile` to bundle the desired models.

### Local FastAPI testing

To run the HTTP API locally:

```bash
source ~/ocr-venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Celery worker helper

If you need to run the Celery queue consumer manually, use:

```bash
./scripts/run_celery_worker.sh
```

## Contributing

Feel free to open issues or submit pull requests as we iterate on the service.

## License

Specify the project license once the service implementation is defined.
