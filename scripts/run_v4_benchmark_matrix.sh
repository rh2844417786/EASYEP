#!/usr/bin/env bash
set -Eeuo pipefail

# Full × prune25 × prune50 evaluation matrix for local Agent-OS, GPQA and
# KuveCodeBench/LiveCodeBench Arrow artifacts. Servers use the same fast V4
# profile: TP=4, decode CUDA graph full (max batch 32), 32 running requests.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FULL_MODEL="${FULL_MODEL:-/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash}"
PRUNE25_MODEL="${PRUNE25_MODEL:-/mnt/docker_data/v4-converted/v4-prune25-keep192}"
PRUNE50_MODEL="${PRUNE50_MODEL:-/mnt/docker_data/v4-converted/v4-prune50-keep128}"
SERVER_PYTHON="${SERVER_PYTHON:-/opt/sglang-v4/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-${SERVER_PYTHON}}"
PORT="${PORT:-60000}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
REPEATS="${REPEATS:-1}"
WORKERS="${WORKERS:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
RUN_ID="${RUN_ID:-v4flash_agentos_gpqa_kuvecodebench_$(date -u +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/easyep_reproduction/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

[[ -x "${SERVER_PYTHON}" ]] || { echo "server Python unavailable: ${SERVER_PYTHON}" >&2; exit 2; }
command -v "${EVAL_PYTHON}" >/dev/null || { echo "evaluation Python unavailable: ${EVAL_PYTHON}" >&2; exit 2; }
[[ "${REPEATS}" =~ ^[1-9][0-9]*$ && "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || { echo "REPEATS and WORKERS must be positive integers" >&2; exit 2; }

mkdir -p "${RESULTS_ROOT}" "${LOG_DIR}"
if "${SERVER_PYTHON}" - "${PORT}" <<'PY'
import sys
from urllib.request import urlopen

try:
    with urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
then
  echo "port ${PORT} already serves SGLang; stop that service before running the model matrix" >&2
  exit 2
fi
declare -a SERVER_PID=()

cleanup() {
  if [[ "${#SERVER_PID[@]}" -gt 0 ]]; then
    kill -TERM -- "-${SERVER_PID[0]}" 2>/dev/null || true
    wait "${SERVER_PID[0]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_CUMEM_HOST_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

start_server() {
  local name="$1" model="$2" log="$3"
  [[ -d "${model}" ]] || { echo "missing ${name} model: ${model}" >&2; return 1; }
  : >"${log}"
  setsid "${SERVER_PYTHON}" -m sglang.launch_server \
    --trust-remote-code --model-path "${model}" --tp 4 --moe-runner-backend marlin \
    --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 --host 127.0.0.1 --port "${PORT}" \
    --context-length 65536 --mem-fraction-static 0.80 --chunked-prefill-size 8192 \
    --max-running-requests 32 --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 32 \
    --watchdog-timeout 1800 --disable-custom-all-reduce --disable-shared-experts-fusion \
    --decode-log-interval 10 --enable-metrics >"${log}" 2>&1 &
  SERVER_PID=("$!")
  local deadline=$((SECONDS + 1800))
  until "${SERVER_PYTHON}" - "${PORT}" <<'PY'
import sys
from urllib.request import urlopen

try:
    with urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2) as response:
        raise SystemExit(response.status != 200)
except Exception:
    raise SystemExit(1)
PY
  do
    kill -0 "${SERVER_PID[0]}" 2>/dev/null || { tail -n 80 "${log}" >&2; return 1; }
    (( SECONDS < deadline )) || { echo "server startup timeout: ${log}" >&2; return 1; }
    sleep 2
  done
  "${SERVER_PYTHON}" "${REPO_ROOT}/scripts/smoke_v4_server.py" --base-url "http://127.0.0.1:${PORT}/v1" \
    >"${RESULTS_ROOT}/${name}/smoke.log" 2>&1
  grep -q 'Capture target decode CUDA graph end' "${log}" || { echo "CUDA graph capture missing after smoke request: ${log}" >&2; return 1; }
}

run_variant() {
  local name model variant_dir
  name="$1"
  model="$2"
  variant_dir="${RESULTS_ROOT}/${name}"
  mkdir -p "${variant_dir}/evaluation"
  start_server "${name}" "${model}" "${variant_dir}/server.log"
  local data
  for data in agent_os gpqa kuvecodebench; do
    echo "[${name}] evaluating ${data}" | tee -a "${variant_dir}/evaluation/${data}.log"
    "${EVAL_PYTHON}" -u "${REPO_ROOT}/evaluation/run_v4_benchmarks.py" \
      --data-name "${data}" --target-path "${variant_dir}/evaluation" \
      --output-file "${variant_dir}/evaluation/${data}.jsonl" \
      --base-url "http://127.0.0.1:${PORT}/v1" --model "${model}" \
      --max-tokens "${MAX_TOKENS}" --workers "${WORKERS}" --repeats "${REPEATS}" \
      --temperature 1.0 --top-p 1.0 --thinking --timeout "${REQUEST_TIMEOUT}" --retries 2 --resume \
      >>"${variant_dir}/evaluation/${data}.log" 2>&1
  done
  cleanup
  SERVER_PID=()
}

run_variant full "${FULL_MODEL}"
run_variant prune25 "${PRUNE25_MODEL}"
run_variant prune50 "${PRUNE50_MODEL}"
echo "Complete: ${RESULTS_ROOT}"
