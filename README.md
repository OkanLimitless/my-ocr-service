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

### Starting the Celery worker

Activate the virtualenv and launch the worker manually:

```bash
source ~/ocr-venv/bin/activate
celery -A celery_app.celery_app worker -Q ocr -l info -c ${CELERY_CONCURRENCY:-4}
```

Or use the convenience wrapper:

```bash
./scripts/run_celery_worker.sh
```

## Contributing

Feel free to open issues or submit pull requests as we iterate on the service.

## License

Specify the project license once the service implementation is defined.
