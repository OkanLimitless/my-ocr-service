#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_celery_worker.sh [--once]
# Optional flags:
#   --once    Run worker in "solo" pool and exit after one task (for debugging)
#
# Environment variables:
#   CELERY_APP_MODULE   Override Celery app module (default: celery_app.celery_app)
#   CELERY_QUEUE        Override queue name (default: ocr)
#   CELERY_LOGLEVEL     Override log level (default: info)
#   CELERY_CONCURRENCY  Override concurrency (default: 4)
#
# Requires the virtualenv created by scripts/runpod_setup.sh (~/ocr-venv).

VENV_PATH="${HOME}/ocr-venv/bin/activate"
if [[ ! -f "$VENV_PATH" ]]; then
  echo "[celery-worker] Virtualenv not found at $VENV_PATH" >&2
  echo "Run scripts/runpod_setup.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_PATH"

APP_MODULE="${CELERY_APP_MODULE:-celery_app.celery_app}"
QUEUE="${CELERY_QUEUE:-ocr}"
LOGLEVEL="${CELERY_LOGLEVEL:-info}"
CONCURRENCY="${CELERY_CONCURRENCY:-4}"

EXTRA_ARGS=()
if [[ ${1:-} == "--once" ]]; then
  EXTRA_ARGS+=("--pool" "solo" "--max-tasks-per-child" "1")
fi

echo "[celery-worker] Starting Celery worker"
echo "  app          = ${APP_MODULE}"
echo "  queue        = ${QUEUE}"
echo "  log level    = ${LOGLEVEL}"
echo "  concurrency  = ${CONCURRENCY}"

exec celery -A "$APP_MODULE" worker -Q "$QUEUE" -l "$LOGLEVEL" -c "$CONCURRENCY" "${EXTRA_ARGS[@]}"
