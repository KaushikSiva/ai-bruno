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

AUDIO_FLAG="${AUDIO_FLAG:-}"
VOICE="${VOICE:-Dominoux}"
WAKE="${WAKE:-hey bruno}"

echo "Starting buddy"
echo "  voice: ${VOICE}"
echo "  wake:  ${WAKE}"

exec python3 "${APP_ROOT}/main.py" \
  --voice "${VOICE}" \
  --wake "${WAKE}" \
  ${AUDIO_FLAG} \
  "$@"
