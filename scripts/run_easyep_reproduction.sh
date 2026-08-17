#!/usr/bin/env bash
set -euo pipefail

# One-command full/prune-25/prune-50 evaluation matrix.
# This wrapper never downloads dependencies, datasets, or checkpoints. Use
# prepare_easyep_evaluation_runtime.sh explicitly if no scoring runtime exists.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FULL_MODEL_PATH="${FULL_MODEL_PATH:-/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash}"
PRUNE25_MODEL_PATH="${PRUNE25_MODEL_PATH:-${REPO_ROOT}/models/v4-prune25-keep192}"
PRUNE50_MODEL_PATH="${PRUNE50_MODEL_PATH:-${REPO_ROOT}/models/v4-prune50-keep128}"
V4_PYTHON="${V4_PYTHON:-/opt/sglang-v4/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-}"
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

supports_math_scoring() {
  "$1" -c \
    "import importlib.util as u, sys; raise SystemExit(0 if sys.version_info >= (3, 10) and (u.find_spec('symeval') or u.find_spec('math_verify')) else 1)" \
    >/dev/null 2>&1
}

resolve_python() {
  local requested="$1"
  if [[ "${requested}" == */* ]]; then
    [[ -x "${requested}" ]] || return 1
    # Keep a virtual environment's bin/python symlink intact. Resolving it
    # reaches the base interpreter and drops the venv site-packages.
    printf '%s\n' "${requested}"
  else
    command -v "${requested}"
  fi
}

select_eval_python() {
  local candidate resolved
  local -a candidates=()
  if [[ -n "${EVAL_PYTHON}" ]]; then
    resolved="$(resolve_python "${EVAL_PYTHON}")" || \
      fail "evaluation Python was not found or is not executable: ${EVAL_PYTHON}"
    supports_math_scoring "${resolved}" || \
      fail "${resolved} requires Python >=3.10 and symeval or math_verify"
    printf '%s\n' "${resolved}"
    return 0
  fi

  candidates+=("/opt/easyep-eval/bin/python")
  command -v python3 >/dev/null 2>&1 && candidates+=("$(command -v python3)")
  command -v python >/dev/null 2>&1 && candidates+=("$(command -v python)")
  candidates+=("${V4_PYTHON}")
  for candidate in "${candidates[@]}"; do
    [[ -x "${candidate}" ]] || continue
    resolved="$(resolve_python "${candidate}")" || continue
    if supports_math_scoring "${resolved}"; then
      printf '%s\n' "${resolved}"
      return 0
    fi
  done
  fail "no local Python >=3.10 has symeval or math_verify; run ALLOW_EVAL_DEP_INSTALL=1 bash scripts/prepare_easyep_evaluation_runtime.sh"
}

for checkpoint_file in \
  "${PRUNE25_MODEL_PATH}/config.json" \
  "${PRUNE25_MODEL_PATH}/model.safetensors.index.json"; do
  [[ -f "${checkpoint_file}" ]] || \
    fail "25%-pruned checkpoint is incomplete; missing ${checkpoint_file}"
done
for checkpoint_file in \
  "${PRUNE50_MODEL_PATH}/config.json" \
  "${PRUNE50_MODEL_PATH}/model.safetensors.index.json"; do
  [[ -f "${checkpoint_file}" ]] || \
    fail "50%-pruned checkpoint is incomplete; missing ${checkpoint_file}"
done
[[ -x "${V4_PYTHON}" ]] || fail "V4 Python is not executable: ${V4_PYTHON}"
# Resolve the evaluation interpreter before runtime validation mutates PATH.
eval_python_path="$(select_eval_python)"
echo "Evaluation Python: ${eval_python_path}"
"${V4_PYTHON}" "${SCRIPT_DIR}/patch_sglang_v4_heterogeneous_experts.py" --check || \
  fail "apply the checked SGLang patch with: ${V4_PYTHON} ${SCRIPT_DIR}/patch_sglang_v4_heterogeneous_experts.py --apply"

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
exec "${eval_python_path}" "${REPO_ROOT}/evaluation/run_reproduction_matrix.py" \
  "${args[@]}" "$@"
