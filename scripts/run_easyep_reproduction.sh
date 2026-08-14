#!/usr/bin/env bash
set -euo pipefail

# One-command full/prune-25/prune-50 evaluation matrix.
# This wrapper never downloads dependencies, datasets, or checkpoints.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FULL_MODEL_PATH="${FULL_MODEL_PATH:-/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash}"
PRUNE25_MODEL_PATH="${PRUNE25_MODEL_PATH:-}"
PRUNE50_MODEL_PATH="${PRUNE50_MODEL_PATH:-}"
V4_PYTHON="${V4_PYTHON:-/opt/sglang-v4/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-python3}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-60000}"
REPEATS="${REPEATS:-5}"
WORKERS="${WORKERS:-1}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/easyep_reproduction}"
RUN_ID="${RUN_ID:-}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -n "${PRUNE25_MODEL_PATH}" ]] || \
  fail "PRUNE25_MODEL_PATH is required and must be a materialized checkpoint with 192/256 experts."
[[ -n "${PRUNE50_MODEL_PATH}" ]] || \
  fail "PRUNE50_MODEL_PATH is required and must be a materialized checkpoint with 128/256 experts."
[[ -x "${V4_PYTHON}" ]] || fail "V4 Python is not executable: ${V4_PYTHON}"
command -v "${EVAL_PYTHON}" >/dev/null 2>&1 || \
  fail "evaluation Python was not found: ${EVAL_PYTHON}"

export MODEL_PATH="${FULL_MODEL_PATH}"
export V4_PYTHON
export V4_GPU_LIST="${GPU_LIST}"
# Source the validator so its CUDA_HOME/PATH/JIT compiler exports are inherited
# by every SGLang server process launched below. The validator is read-only and
# explicitly rejects missing runtime components instead of installing them.
export V4_RUNTIME_VALIDATOR_LIB_ONLY=1
# shellcheck source=validate_v4_flash_runtime.sh
source "${SCRIPT_DIR}/validate_v4_flash_runtime.sh"
unset V4_RUNTIME_VALIDATOR_LIB_ONLY
validate_v4_flash_runtime

eval_python_path="$(command -v "${EVAL_PYTHON}")"
args=(
  --full-model "${FULL_MODEL_PATH}"
  --prune25-model "${PRUNE25_MODEL_PATH}"
  --prune50-model "${PRUNE50_MODEL_PATH}"
  --server-python "${V4_PYTHON}"
  --eval-python "${eval_python_path}"
  --gpu-list "${GPU_LIST}"
  --tp "${TP_SIZE}"
  --port "${PORT}"
  --repeats "${REPEATS}"
  --workers "${WORKERS}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --max-tokens "${MAX_TOKENS}"
  --context-length "${CONTEXT_LENGTH}"
  --results-root "${RESULTS_ROOT}"
)
[[ -n "${RUN_ID}" ]] && args+=(--run-id "${RUN_ID}")
[[ "${CONTINUE_ON_ERROR:-0}" == "1" ]] && args+=(--continue-on-error)
[[ "${DRY_RUN:-0}" == "1" ]] && args+=(--dry-run)
[[ "${ALLOW_HASH_ROUTED_PRUNED_CHECKPOINTS:-0}" == "1" ]] && \
  args+=(--allow-hash-routed-pruned-checkpoints)

exec "${eval_python_path}" "${REPO_ROOT}/evaluation/run_reproduction_matrix.py" \
  "${args[@]}" "$@"
