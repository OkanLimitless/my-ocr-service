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

RunPod Serverless invokes `serverless_handler.handler`. A minimal payload looks like:

```json
{
  "task": "tasks.ocr_document",
  "payload": {
    "url": "https://example.com/sample.png"
  }
}
```

The handler also accepts `base64` plus an optional `suffix` field.

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
