#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/runpod_setup.sh /path/to/requirements.txt
# Example: ./scripts/runpod_setup.sh /workspace/lernio/apps/worker/requirements.txt

if [[ ${EUID} -ne 0 ]]; then
  echo "[runpod-setup] Please re-run this script with sudo or root privileges." >&2
  exit 1
fi

REQ_FILE=${1:-}
if [[ -z "$REQ_FILE" ]]; then
  echo "[runpod-setup] Provide the path to requirements.txt as the first argument." >&2
  exit 1
fi

if [[ ! -f "$REQ_FILE" ]]; then
  echo "[runpod-setup] requirements.txt not found: $REQ_FILE" >&2
  exit 1
fi

apt-get update
apt-get install -y \
  build-essential pkg-config swig \
  ffmpeg libgomp1 libglib2.0-0 libsm6 \
  libxext6 libxrender1

su - "${SUDO_USER:-root}" -c "python3 -m venv ~/ocr-venv"

# shellcheck disable=SC1090
su - "${SUDO_USER:-root}" -c "source ~/ocr-venv/bin/activate && \ 
  pip install --upgrade pip && \ 
  pip install -r '$REQ_FILE'"

printf '\n[runpod-setup] Complete. Activate the venv with:\n'
printf '  source ~/ocr-venv/bin/activate\n'
