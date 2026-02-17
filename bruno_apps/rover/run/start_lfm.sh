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

LFM2_VL_PORT="${LFM2_VL_PORT:-8081}"
LFM2_VL_CTX="${LFM2_VL_CTX:-1024}"

echo "Starting local LFM2-VL server"
echo "  port: ${LFM2_VL_PORT}"
echo "  ctx:  ${LFM2_VL_CTX}"

export LFM2_VL_PORT LFM2_VL_CTX
exec "${APP_ROOT}/run/run_lfm2_vl_server.sh"
