#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BRUNO_CAMERA_URL="${BRUNO_CAMERA_URL:-http://127.0.0.1:8080/}"
VLM_LOCAL_BASE="${VLM_LOCAL_BASE:-http://127.0.0.1:8081/v1}"
VLM_LOCAL_TIMEOUT_MS="${VLM_LOCAL_TIMEOUT_MS:-4000}"
VLM_MAX_LATENCY_MS="${VLM_MAX_LATENCY_MS:-4500}"
VLM_INTERVAL="${VLM_INTERVAL:-3.0}"

echo "Starting rover"
echo "  camera url:          ${BRUNO_CAMERA_URL}"
echo "  vlm local base:      ${VLM_LOCAL_BASE}"
echo "  vlm timeout ms:      ${VLM_LOCAL_TIMEOUT_MS}"
echo "  vlm max latency ms:  ${VLM_MAX_LATENCY_MS}"
echo "  vlm interval:        ${VLM_INTERVAL}"

export BRUNO_CAMERA_URL

exec python3 "${REPO_ROOT}/bruno_cam_rover.py" \
  --mode builtin \
  --vlm-provider local \
  --vlm-local-base "${VLM_LOCAL_BASE}" \
  --vlm-local-timeout-ms "${VLM_LOCAL_TIMEOUT_MS}" \
  --vlm-max-latency-ms "${VLM_MAX_LATENCY_MS}" \
  --vlm-interval "${VLM_INTERVAL}" \
  "$@"
