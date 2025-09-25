#!/usr/bin/env bash
set -euo pipefail

# Usage: sudo ./scripts/runpod_setup.sh [path/to/requirements.txt]
# If no path is provided the repository requirements.txt is used.

if [[ ${EUID} -ne 0 ]]; then
  echo "[runpod-setup] Please re-run this script with sudo or root privileges." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_REQ_FILE="$REPO_ROOT/requirements.txt"
REQ_FILE="${1:-$DEFAULT_REQ_FILE}"

if [[ ! -f "$REQ_FILE" ]]; then
  echo "[runpod-setup] requirements.txt not found: $REQ_FILE" >&2
  exit 1
fi

apt-get update
apt-get install -y \
  build-essential pkg-config swig \
  ffmpeg libgomp1 libglib2.0-0 libsm6 \
  libxext6 libxrender1

TARGET_USER="${SUDO_USER:-root}"

su - "$TARGET_USER" -c "python3 -m venv ~/ocr-venv"

# shellcheck disable=SC1090
su - "$TARGET_USER" -c "source ~/ocr-venv/bin/activate && \ 
  pip install --upgrade pip && \ 
  pip install -r '$REQ_FILE'"

printf '\n[runpod-setup] Complete. Activate the venv with:\n'
printf '  source ~/ocr-venv/bin/activate\n'
printf '\n[runpod-setup] Installed requirements from: %s\n' "$REQ_FILE"
