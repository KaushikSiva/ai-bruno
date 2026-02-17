#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

STREAM_SOURCE="${STREAM_SOURCE:-/dev/video0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

echo "Starting MJPEG stream server"
echo "  source: ${STREAM_SOURCE}"
echo "  host:   ${HOST}"
echo "  port:   ${PORT}"

export STREAM_SOURCE HOST PORT
exec python3 "${REPO_ROOT}/bruno_apps/rover/live_stream.py"
