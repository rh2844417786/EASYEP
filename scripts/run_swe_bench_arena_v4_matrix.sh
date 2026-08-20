#!/usr/bin/env bash
set -Eeuo pipefail

# Run the official SWE-bench arena image against full, 25%-pruned, and
# 50%-pruned V4-Flash checkpoints. The server runs in wth333 on GPUs 4-7; the
# benchmark runs on the host with host networking because this machine has no
# usable Docker bridge network.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SGLANG_CONTAINER="${SGLANG_CONTAINER:-wth333}"
BENCHMARK_IMAGE="${BENCHMARK_IMAGE:-swe_bench_arena:0.3.2}"
SERVER_PYTHON="${SERVER_PYTHON:-/opt/sglang-v4/bin/python}"
FULL_MODEL="${FULL_MODEL:-/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash}"
PRUNE25_MODEL="${PRUNE25_MODEL:-/mnt/docker_data/v4-converted/v4-prune25-keep192}"
PRUNE50_MODEL="${PRUNE50_MODEL:-/mnt/docker_data/v4-converted/v4-prune50-keep128}"
PORT="${PORT:-8216}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
CONCURRENCIES="${CONCURRENCIES:-20}"
RUN_ID="${RUN_ID:-swe_bench_arena_v4_$(date -u +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/swe_bench_arena/${RUN_ID}}"

command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }
docker inspect "${SGLANG_CONTAINER}" >/dev/null 2>&1 || { echo "missing container: ${SGLANG_CONTAINER}" >&2; exit 2; }
docker image inspect "${BENCHMARK_IMAGE}" >/dev/null 2>&1 || { echo "missing benchmark image: ${BENCHMARK_IMAGE}" >&2; exit 2; }
for concurrency in ${CONCURRENCIES}; do
  [[ "${concurrency}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid concurrency: ${concurrency}" >&2; exit 2; }
done

mkdir -p "${RESULTS_ROOT}"
SERVER_PID=""

server_ready() {
  docker exec "${SGLANG_CONTAINER}" "${SERVER_PYTHON}" -c \
    "from urllib.request import urlopen; urlopen('http://127.0.0.1:${PORT}/health', timeout=2).close()" \
    >/dev/null 2>&1
}

if server_ready; then
  echo "port ${PORT} already serves SGLang; stop that service before running this matrix" >&2
  exit 2
fi

stop_server() {
  if [[ -n "${SERVER_PID}" ]]; then
    docker exec "${SGLANG_CONTAINER}" bash -lc "kill -TERM -- -${SERVER_PID} 2>/dev/null || true" || true
    for _ in $(seq 1 30); do
      docker exec "${SGLANG_CONTAINER}" kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    docker exec "${SGLANG_CONTAINER}" bash -lc "kill -KILL -- -${SERVER_PID} 2>/dev/null || true" || true
    SERVER_PID=""
  fi
}
trap stop_server EXIT INT TERM

start_server() {
  local variant="$1"
  local model="$2"
  local variant_dir="$3"
  local server_log="${variant_dir}/server.log"
  [[ -n "${SERVER_PID}" ]] && stop_server
  docker exec "${SGLANG_CONTAINER}" test -d "${model}" || { echo "missing ${variant} model: ${model}" >&2; return 1; }
  : >"${server_log}"
  SERVER_PID="$(docker exec "${SGLANG_CONTAINER}" bash -lc "cd '${REPO_ROOT}'; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' PYTHONUNBUFFERED=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_CUMEM_HOST_ENABLE=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_BLOCKING_WAIT=1; setsid '${SERVER_PYTHON}' -m sglang.launch_server --trust-remote-code --model-path '${model}' --tp 4 --moe-runner-backend marlin --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 --served-model-name deepseek-v4-flash --host 0.0.0.0 --port '${PORT}' --context-length 262144 --mem-fraction-static 0.80 --chunked-prefill-size 8192 --max-running-requests 32 --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 32 --watchdog-timeout 1800 --disable-custom-all-reduce --disable-shared-experts-fusion --decode-log-interval 10 --enable-metrics > '${server_log}' 2>&1 & echo \$!")"
  local deadline=$((SECONDS + 1800))
  until server_ready; do
    docker exec "${SGLANG_CONTAINER}" kill -0 "${SERVER_PID}" 2>/dev/null || {
      tail -n 100 "${server_log}" >&2
      return 1
    }
    ((SECONDS < deadline)) || { echo "${variant} server startup timeout" >&2; return 1; }
    sleep 2
  done
}

run_variant() {
  local variant="$1"
  local model="$2"
  local variant_dir="${RESULTS_ROOT}/${variant}"
  mkdir -p "${variant_dir}"
  start_server "${variant}" "${model}" "${variant_dir}"
  for concurrency in ${CONCURRENCIES}; do
    local benchmark_dir="${variant_dir}/c_${concurrency}"
    mkdir -p "${benchmark_dir}"
    echo "[${variant}] concurrency=${concurrency}"
    docker run --rm --network host -u 0:0 \
      --name "easyep-swe-${RUN_ID}-${variant}-c${concurrency}" \
      -e URL="http://127.0.0.1:${PORT}" -e MODEL=deepseek-v4-flash -e CONCURRENCY="${concurrency}" \
      -v "${benchmark_dir}:/data/output" "${BENCHMARK_IMAGE}" \
      >"${benchmark_dir}/benchmark.log" 2>&1
  done
  stop_server
}

run_variant full "${FULL_MODEL}"
run_variant prune25 "${PRUNE25_MODEL}"
run_variant prune50 "${PRUNE50_MODEL}"
echo "Complete: ${RESULTS_ROOT}"
