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

MODE="${MODE:-external}"
VOICE="${VOICE:-Dominoux}"
SCAN_SPEED="${SCAN_SPEED:-1.5}"
AUDIO_FLAG="${AUDIO_FLAG:-}"
HEADLESS_FLAG="${HEADLESS_FLAG---headless}"
INVERT_CAMERA_FLAG="${INVERT_CAMERA_FLAG:-}"
INVERT_VERTICAL_FLAG="${INVERT_VERTICAL_FLAG:-}"

echo "Starting face follower"
echo "  mode:       ${MODE}"
echo "  voice:      ${VOICE}"
echo "  scan speed: ${SCAN_SPEED}"

exec python3 "${APP_ROOT}/main.py" \
  --mode "${MODE}" \
  --voice "${VOICE}" \
  --scan-speed "${SCAN_SPEED}" \
  ${AUDIO_FLAG} \
  ${HEADLESS_FLAG} \
  ${INVERT_CAMERA_FLAG} \
  ${INVERT_VERTICAL_FLAG} \
  "$@"
