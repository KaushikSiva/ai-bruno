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

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
PORT="${LFM2_VL_PORT:-8080}"
CTX="${LFM2_VL_CTX:-2048}"
HF_REPO="${LFM2_VL_HF_REPO:-LiquidAI/LFM2-VL-450M-GGUF}"
HF_FILE="${LFM2_VL_HF_FILE:-Q4_0}"

SERVER_BIN="$LLAMA_DIR/build/bin/llama-server"
if [ ! -x "$SERVER_BIN" ]; then
  echo "ERROR: llama-server not found at $SERVER_BIN"
  echo "Run: bruno_apps/rover/run/setup_lfm2_vl_pi.sh"
  exit 1
fi

echo "Starting llama-server on 0.0.0.0:$PORT"
echo "Model: $HF_REPO:$HF_FILE"
exec "$SERVER_BIN" -hf "$HF_REPO:$HF_FILE" -c "$CTX" --port "$PORT"
