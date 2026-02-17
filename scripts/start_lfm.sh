#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LFM2_VL_PORT="${LFM2_VL_PORT:-8081}"
LFM2_VL_CTX="${LFM2_VL_CTX:-1024}"

echo "Starting local LFM2-VL server"
echo "  port: ${LFM2_VL_PORT}"
echo "  ctx:  ${LFM2_VL_CTX}"

export LFM2_VL_PORT LFM2_VL_CTX
exec "${REPO_ROOT}/scripts/run_lfm2_vl_server.sh"
