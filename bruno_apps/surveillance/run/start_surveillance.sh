#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

MODE="${MODE:-builtin}"
AUDIO_FLAG="${AUDIO_FLAG:-}"
VOICE="${VOICE:-Dominoux}"
CAPTION_BACKEND="${CAPTION_BACKEND:-local}"

echo "Starting surveillance"
echo "  mode:            ${MODE}"
echo "  caption backend: ${CAPTION_BACKEND}"
echo "  voice:           ${VOICE}"

exec python3 "${APP_ROOT}/main.py" \
  --mode "${MODE}" \
  --voice "${VOICE}" \
  --caption-backend "${CAPTION_BACKEND}" \
  ${AUDIO_FLAG} \
  "$@"
