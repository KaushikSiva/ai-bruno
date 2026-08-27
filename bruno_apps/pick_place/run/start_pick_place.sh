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

PROMPT="${PROMPT:-${1:-pick up the blue bottle and put it on the left}}"
VLM_LOCAL_BASE="${VLM_LOCAL_BASE:-http://127.0.0.1:8081/v1}"
VLM_LOCAL_MODEL="${VLM_LOCAL_MODEL:-gemma3}"
PP_STANDOFF_CM="${PP_STANDOFF_CM:-18}"
PP_SPEED="${PP_SPEED:-25}"
MODE="${MODE:-builtin}"

echo "Starting pick and place"
echo "  prompt:      ${PROMPT}"
echo "  mode:        ${MODE}"
echo "  vlm base:    ${VLM_LOCAL_BASE}"
echo "  vlm model:   ${VLM_LOCAL_MODEL}"
echo "  standoff cm: ${PP_STANDOFF_CM}"

exec python3 "${APP_ROOT}/main.py" \
  --prompt "${PROMPT}" \
  --mode "${MODE}" \
  --vlm-base "${VLM_LOCAL_BASE}" \
  --vlm-model "${VLM_LOCAL_MODEL}" \
  --standoff "${PP_STANDOFF_CM}" \
  --speed "${PP_SPEED}"
