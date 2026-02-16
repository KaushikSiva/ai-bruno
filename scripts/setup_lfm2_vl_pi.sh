#!/usr/bin/env bash
set -euo pipefail

# Setup script for Raspberry Pi:
# - installs build deps
# - clones/builds llama.cpp
# - writes a helper run script for LFM2-VL-450M server

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
PORT="${LFM2_VL_PORT:-8080}"
CTX="${LFM2_VL_CTX:-2048}"
THREADS="${LFM2_VL_THREADS:-4}"
HF_REPO="${LFM2_VL_HF_REPO:-LiquidAI/LFM2-VL-450M-GGUF}"
HF_FILE="${LFM2_VL_HF_FILE:-Q4_0}"
RUN_SCRIPT="${RUN_SCRIPT:-$(pwd)/scripts/run_lfm2_vl_server.sh}"

echo "[1/5] Checking base tools..."
if ! command -v git >/dev/null 2>&1; then
  echo "git not found. Installing dependencies with apt..."
  sudo apt-get update
  sudo apt-get install -y git cmake build-essential
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found. Installing dependencies with apt..."
  sudo apt-get update
  sudo apt-get install -y cmake build-essential
fi

echo "[2/5] Fetching llama.cpp into $LLAMA_DIR ..."
if [ -d "$LLAMA_DIR/.git" ]; then
  git -C "$LLAMA_DIR" pull --ff-only
else
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi

echo "[3/5] Building llama.cpp ..."
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build"
cmake --build "$LLAMA_DIR/build" --config Release -j "$THREADS"

SERVER_BIN="$LLAMA_DIR/build/bin/llama-server"
if [ ! -x "$SERVER_BIN" ]; then
  echo "ERROR: llama-server binary not found at $SERVER_BIN"
  exit 1
fi

echo "[4/5] Writing helper script: $RUN_SCRIPT"
mkdir -p "$(dirname "$RUN_SCRIPT")"
cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail

LLAMA_DIR="\${LLAMA_DIR:-$LLAMA_DIR}"
PORT="\${LFM2_VL_PORT:-$PORT}"
CTX="\${LFM2_VL_CTX:-$CTX}"
HF_REPO="\${LFM2_VL_HF_REPO:-$HF_REPO}"
HF_FILE="\${LFM2_VL_HF_FILE:-$HF_FILE}"

SERVER_BIN="\$LLAMA_DIR/build/bin/llama-server"
if [ ! -x "\$SERVER_BIN" ]; then
  echo "ERROR: llama-server not found at \$SERVER_BIN"
  exit 1
fi

echo "Starting llama-server on 0.0.0.0:\$PORT"
echo "Model: \$HF_REPO:\$HF_FILE"
exec "\$SERVER_BIN" -hf "\$HF_REPO:\$HF_FILE" -c "\$CTX" --port "\$PORT"
EOF
chmod +x "$RUN_SCRIPT"

echo "[5/5] Done."
echo
echo "Next steps:"
echo "  1) Start local VLM server:"
echo "     $RUN_SCRIPT"
echo
echo "  2) Run Bruno rover with local-first fallback:"
echo "     python3 bruno_cam_rover.py --mode external --show \\"
echo "       --vlm-provider auto \\"
echo "       --vlm-local-base http://127.0.0.1:${PORT}/v1 \\"
echo "       --vlm-remote-base http://<MAC_IP>:1234/v1"
