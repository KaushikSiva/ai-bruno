#!/usr/bin/env bash
# Interactive teleop. Defaults to simulation; TARGET=real drives the robot.
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

PYTHON="${PYTHON:-python3}"
if [ -x "${REPO_ROOT}/venv/bin/python" ]; then
  PYTHON="${REPO_ROOT}/venv/bin/python"
fi

TARGET="${TARGET:-sim}"
ROBOT_URL="${ROBOT_URL:-}"
MIRROR_URL="${MIRROR_URL:-}"

echo "Teleop target: ${TARGET}${ROBOT_URL:+ via ${ROBOT_URL}}${MIRROR_URL:+ | mirroring to ${MIRROR_URL}}"

ARGS=(--target "${TARGET}")
if [ -n "${ROBOT_URL}" ]; then
  ARGS+=(--robot-url "${ROBOT_URL}")
fi
if [ -n "${MIRROR_URL}" ]; then
  ARGS+=(--mirror-url "${MIRROR_URL}")
fi

exec "${PYTHON}" "${APP_ROOT}/main.py" keys "${ARGS[@]}" "$@"
