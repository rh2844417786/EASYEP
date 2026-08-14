#!/usr/bin/env bash
set -euo pipefail

# Four-GPU DeepSeek-V4-Flash feasibility test.
#
# This script intentionally fixes the physical GPU selection to 4,5,6,7 and
# TP to 4. It is a diagnostic only: the verified H100 baseline remains TP=8.
#
# Defaults target the in-place Docker runtime prepared for DeepSeek-V4-Flash.
# Override any value by exporting it before running this script.
MODEL_PATH="${MODEL_PATH:-/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash}"
V4_PYTHON="${V4_PYTHON:-/opt/sglang-v4/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly GPU_LIST="4,5,6,7"
readonly TP_SIZE="4"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-60000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
STREAM_LOGS="${STREAM_LOGS:-0}"
SERVER_PID=""
LOG_TAIL_PID=""
SMOKE_OUTPUT=""
NVCC_VERSION=""

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

cleanup() {
  if [[ -n "${LOG_TAIL_PID}" ]] && kill -0 "${LOG_TAIL_PID}" 2>/dev/null; then
    kill "${LOG_TAIL_PID}" 2>/dev/null || true
    wait "${LOG_TAIL_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Stopping diagnostic server (PID ${SERVER_PID})..."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Validate the existing runtime before any model load.  The validator performs
# no package/model download and exports the validated CUDA compiler plus
# offline Hugging Face/Transformers settings into this process.
export V4_RUNTIME_VALIDATOR_LIB_ONLY=1
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/validate_v4_flash_runtime.sh"
unset V4_RUNTIME_VALIDATOR_LIB_ONLY
validate_v4_flash_runtime

SGLANG_VERSION="${V4_VALIDATED_SGLANG_VERSION}"
SGLANG_MODULE="${V4_VALIDATED_SGLANG_MODULE}"
NVCC_VERSION="${V4_VALIDATED_NVCC_VERSION}"
GPU_INVENTORY="$(nvidia-smi --query-gpu=index,name,memory.free \
  --format=csv,noheader,nounits)" || fail "nvidia-smi failed"

if "${V4_PYTHON}" - "${PORT}" <<'PY'
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
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_STARTED_AT="$(date -Iseconds 2>/dev/null || date)"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || printf 'unknown')"
LOG_FILE="${LOG_DIR}/v4_gpus_4_7_${RUN_ID}.log"
SUMMARY_FILE="${LOG_DIR}/v4_gpus_4_7_${RUN_ID}_summary.txt"

write_summary() {
  local result="$1"
  local server_status="${2:-not_available}"
  local smoke_status="${3:-not_run}"
  local ended_at
  ended_at="$(date -Iseconds 2>/dev/null || date)"

  {
    echo "DeepSeek-V4-Flash four-GPU test summary"
    echo "========================================"
    echo "Result: ${result}"
    echo "Started: ${RUN_STARTED_AT}"
    echo "Ended: ${ended_at}"
    echo "Git commit: ${GIT_COMMIT}"
    echo "Server exit status: ${server_status}"
    echo "Smoke status: ${smoke_status}"
    echo "Full log: ${LOG_FILE}"
    echo
    echo "Runtime"
    echo "-------"
    echo "Python: ${V4_PYTHON}"
    echo "SGLang: ${SGLANG_VERSION}"
    echo "NVCC: ${NVCC_VERSION} (${DG_JIT_NVCC_COMPILER})"
    echo "CUDA_HOME: ${CUDA_HOME}"
    echo "Model: ${MODEL_PATH}"
    echo "Physical GPUs: ${GPU_LIST}"
    echo "TP: ${TP_SIZE}"
    echo "Context length: ${CONTEXT_LENGTH}"
    echo "Memory fraction static: ${MEM_FRACTION_STATIC}"
    echo "Custom AllReduce: disabled"
    echo "HF Hub offline: ${HF_HUB_OFFLINE}"
    echo "Transformers offline: ${TRANSFORMERS_OFFLINE}"
    echo "NCCL_IB_DISABLE: ${NCCL_IB_DISABLE}"
    echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
    echo "NCCL_CUMEM_HOST_ENABLE: ${NCCL_CUMEM_HOST_ENABLE}"
    echo
    echo "GPU snapshot"
    echo "------------"
    nvidia-smi -i "${GPU_LIST}" \
      --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader || true
    echo
    echo "Host/container memory"
    echo "---------------------"
    free -h || true
    if [[ -r /sys/fs/cgroup/memory.events ]]; then
      echo
      echo "/sys/fs/cgroup/memory.events"
      cat /sys/fs/cgroup/memory.events || true
    fi
    if [[ -n "${SMOKE_OUTPUT}" ]]; then
      echo
      echo "Smoke output"
      echo "------------"
      printf '%s\n' "${SMOKE_OUTPUT}"
    fi
    echo
    echo "Key log lines (last 60 matches)"
    echo "-------------------------------"
    grep -Eai \
      'traceback|error|exception|failed|killed|out of memory|oom|ninja: build stopped|nccl warn|server is ready|loading.*weight|load.*model|memory' \
      "${LOG_FILE}" | tail -n 60 || true
    echo
    echo "Last 40 full-log lines"
    echo "----------------------"
    tail -n 40 "${LOG_FILE}" || true
  } >"${SUMMARY_FILE}" 2>&1

  echo "Summary report: ${SUMMARY_FILE}"
}

echo "WARNING: this is an unverified TP=4 feasibility test, not the 8-GPU baseline."
echo "Physical GPUs: ${GPU_LIST} (inside SGLang they appear as cuda:0..3)"
echo "Python: ${V4_PYTHON}"
echo "SGLang: ${SGLANG_VERSION} (${SGLANG_MODULE})"
echo "NVCC: ${NVCC_VERSION} (${DG_JIT_NVCC_COMPILER})"
echo "CUDA_HOME: ${CUDA_HOME}"
echo "Model: ${MODEL_PATH}"
echo "Log: ${LOG_FILE}"
echo "Summary: ${SUMMARY_FILE}"
echo "Selected GPU inventory:"
awk -F, '$1 ~ /^[[:space:]]*[4567][[:space:]]*$/ { print "  " $0 }' \
  <<<"${GPU_INVENTORY}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# The standalone four-GPU all-reduce test succeeds with this single-node
# Docker configuration.  Keep NCCL on loopback for bootstrap and avoid the
# container NUMA/cuMem-host path that can hang during communicator setup.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

echo "NCCL: IB_DISABLE=${NCCL_IB_DISABLE}, SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}, CUMEM_HOST_ENABLE=${NCCL_CUMEM_HOST_ENABLE}"
echo "Custom AllReduce: disabled (v0.5.16 tvm-ffi JIT compile workaround)"
echo "Downloads: disabled; HF_HUB_OFFLINE=${HF_HUB_OFFLINE}, TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"

"${V4_PYTHON}" -m sglang.launch_server \
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
  --disable-custom-all-reduce \
  "$@" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

echo "Waiting up to ${STARTUP_TIMEOUT}s for /v1/models (PID ${SERVER_PID})..."
if [[ "${STREAM_LOGS}" == "1" ]]; then
  echo "Streaming SGLang output from ${LOG_FILE}:"
  tail -n +1 -F "${LOG_FILE}" &
  LOG_TAIL_PID=$!
else
  echo "Full log streaming is disabled; set STREAM_LOGS=1 to enable it."
fi

wait_started=$SECONDS
next_progress=$((SECONDS + 60))
deadline=$((SECONDS + STARTUP_TIMEOUT))
ready=0
while (( SECONDS < deadline )); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    server_status=0
    wait "${SERVER_PID}" 2>/dev/null || server_status=$?
    SERVER_PID=""
    if [[ -n "${LOG_TAIL_PID}" ]] && kill -0 "${LOG_TAIL_PID}" 2>/dev/null; then
      kill "${LOG_TAIL_PID}" 2>/dev/null || true
      wait "${LOG_TAIL_PID}" 2>/dev/null || true
    fi
    LOG_TAIL_PID=""
    write_summary "SERVER_EXITED_BEFORE_READY" "${server_status}" "not_run"
    echo "SGLang exited before becoming ready; send the summary TXT above." >&2
    exit 1
  fi

  if "${V4_PYTHON}" - "http://127.0.0.1:${PORT}/v1/models" <<'PY'
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
  if (( SECONDS >= next_progress )); then
    echo "Still waiting: $((SECONDS - wait_started))s elapsed; GPU memory:"
    nvidia-smi -i "${GPU_LIST}" \
      --query-gpu=index,memory.used,memory.free \
      --format=csv,noheader || true
    next_progress=$((SECONDS + 60))
  fi
  sleep 10
done

if [[ "${ready}" != "1" ]]; then
  write_summary "STARTUP_TIMEOUT" "still_running" "not_run"
  echo "SGLang did not become ready within ${STARTUP_TIMEOUT}s; send the summary TXT above." >&2
  exit 1
fi

echo "Service is ready; sending the smoke request..."
smoke_status=0
SMOKE_OUTPUT="$("${V4_PYTHON}" "${REPO_ROOT}/scripts/smoke_v4_server.py" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --timeout 600 2>&1)" || smoke_status=$?
printf '%s\n' "${SMOKE_OUTPUT}"
if [[ "${smoke_status}" != "0" ]]; then
  write_summary "SMOKE_FAILED" "running" "${smoke_status}"
  echo "Smoke request failed; send the summary TXT above." >&2
  exit 1
fi

write_summary "PASSED" "running" "0"
echo "Four-GPU smoke test passed. Full server log: ${LOG_FILE}"
