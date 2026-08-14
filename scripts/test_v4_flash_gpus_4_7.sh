#!/usr/bin/env bash
set -euo pipefail

# Four-GPU DeepSeek-V4-Flash feasibility test.
#
# This script intentionally fixes the physical GPU selection to 4,5,6,7 and
# TP to 4. It is a diagnostic only: the verified H100 baseline remains TP=8.
#
# Required:
#   MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash
# Optional:
#   PORT=60000 CONTEXT_LENGTH=32768 STARTUP_TIMEOUT=3600

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "ERROR: MODEL_PATH is required" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly GPU_LIST="4,5,6,7"
readonly TP_SIZE="4"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-60000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
SERVER_PID=""

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Stopping diagnostic server (PID ${SERVER_PID})..."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ -f "${MODEL_PATH}/config.json" ]] || \
  fail "${MODEL_PATH}/config.json was not found"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi was not found"

GPU_INVENTORY="$(nvidia-smi --query-gpu=index,name,memory.free \
  --format=csv,noheader,nounits)" || fail "nvidia-smi failed"
for gpu_index in 4 5 6 7; do
  if ! awk -F, -v expected="${gpu_index}" \
    '{ index=$1; gsub(/[[:space:]]/, "", index); if (index == expected) found=1 } END { exit !found }' \
    <<<"${GPU_INVENTORY}"; then
    fail "physical GPU ${gpu_index} was not reported by nvidia-smi"
  fi
done

if ! SGLANG_VERSION="$(python3 -c \
  'from importlib.metadata import version; print(version("sglang"))' 2>/dev/null)"; then
  fail "the SGLang Python package is not installed in this environment"
fi
if ! SGLANG_MODULE="$(python3 -c \
  'import sglang; print(sglang.__file__ or "")' 2>/dev/null)"; then
  fail "the installed SGLang package cannot be imported"
fi
[[ -n "${SGLANG_MODULE}" ]] || \
  fail "Python resolved sglang only as a namespace; install the SGLang package"
case "${SGLANG_MODULE}" in
  "${REPO_ROOT}"/sglang/*)
    fail "Python resolved the repository's legacy SGLang copy: ${SGLANG_MODULE}"
    ;;
esac

if python3 - "${PORT}" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(1)
    raise SystemExit(sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) != 0)
PY
then
  fail "port ${PORT} is already in use; choose another PORT"
fi

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/v4_gpus_4_7_$(date +%Y%m%d_%H%M%S).log"

echo "WARNING: this is an unverified TP=4 feasibility test, not the 8-GPU baseline."
echo "Physical GPUs: ${GPU_LIST} (inside SGLang they appear as cuda:0..3)"
echo "SGLang: ${SGLANG_VERSION} (${SGLANG_MODULE})"
echo "Model: ${MODEL_PATH}"
echo "Log: ${LOG_FILE}"
echo "Selected GPU inventory:"
awk -F, '$1 ~ /^[[:space:]]*[4567][[:space:]]*$/ { print "  " $0 }' \
  <<<"${GPU_INVENTORY}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"

python3 -m sglang.launch_server \
  --trust-remote-code \
  --model-path "${MODEL_PATH}" \
  --tp "${TP_SIZE}" \
  --moe-runner-backend marlin \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --host "${HOST}" \
  --port "${PORT}" \
  --disable-cuda-graph \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --context-length "${CONTEXT_LENGTH}" \
  --max-running-requests 1 \
  "$@" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

echo "Waiting up to ${STARTUP_TIMEOUT}s for /v1/models (PID ${SERVER_PID})..."
deadline=$((SECONDS + STARTUP_TIMEOUT))
ready=0
while (( SECONDS < deadline )); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
    echo "SGLang exited before becoming ready. Last 120 log lines:" >&2
    tail -n 120 "${LOG_FILE}" >&2 || true
    exit 1
  fi

  if python3 - "http://127.0.0.1:${PORT}/v1/models" <<'PY'
import sys
from urllib.request import urlopen

try:
    with urlopen(sys.argv[1], timeout=2) as response:
        raise SystemExit(response.status != 200)
except Exception:
    raise SystemExit(1)
PY
  then
    ready=1
    break
  fi
  sleep 10
done

if [[ "${ready}" != "1" ]]; then
  echo "SGLang did not become ready within ${STARTUP_TIMEOUT}s. Last 120 log lines:" >&2
  tail -n 120 "${LOG_FILE}" >&2 || true
  exit 1
fi

echo "Service is ready; sending the smoke request..."
if ! python3 "${REPO_ROOT}/scripts/smoke_v4_server.py" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --timeout 600; then
  echo "Smoke request failed. Last 120 server log lines:" >&2
  tail -n 120 "${LOG_FILE}" >&2 || true
  exit 1
fi

echo "Four-GPU smoke test passed. Full server log: ${LOG_FILE}"
