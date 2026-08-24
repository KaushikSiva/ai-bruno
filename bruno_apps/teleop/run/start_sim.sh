#!/usr/bin/env bash
# Start the MuJoCo simulation bridge, with the viewer unless HEADLESS is set.
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

PYTHON="${PYTHON:-python3}"
if [ -x "${REPO_ROOT}/venv/bin/python" ]; then
  PYTHON="${REPO_ROOT}/venv/bin/python"
fi

VIEWER_FLAG="--mujoco-viewer"
if [ -n "${HEADLESS:-}" ]; then
  VIEWER_FLAG=""
fi

if [ ! -f "${REPO_ROOT}/third_party/masterpi-mujoco/masterpi.xml" ]; then
  echo "MasterPi model missing. Run: git submodule update --init --recursive" >&2
  exit 1
fi

echo "Starting the MuJoCo bridge on http://127.0.0.1:${BRUNO_BRIDGE_PORT:-8091} (starts disarmed)"

cd "${REPO_ROOT}"
exec "${PYTHON}" -m bruno_core.vla.server --driver mujoco ${VIEWER_FLAG} "$@"
