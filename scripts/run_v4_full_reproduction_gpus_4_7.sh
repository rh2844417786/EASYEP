#!/usr/bin/env bash
set -Eeuo pipefail

# End-to-end DeepSeek-V4-Flash EASY-EP reproduction on physical GPUs 4..7.
# It converts local HF weights once, materializes/validates both pruned models,
# and executes the full/prune25/prune50 evaluation matrix. No downloads or
# dependency installation are performed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FULL_MODEL_PATH="${FULL_MODEL_PATH:-/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash}"
V4_INFERENCE_DIR="${V4_INFERENCE_DIR:-${FULL_MODEL_PATH}/inference}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/docker_data/v4-converted}"
CONVERTED_CKPT_PATH="${CONVERTED_CKPT_PATH:-${ARTIFACT_ROOT}/mp4-fp4}"
PRUNE25_MODEL_PATH="${PRUNE25_MODEL_PATH:-${ARTIFACT_ROOT}/v4-prune25-keep192}"
PRUNE50_MODEL_PATH="${PRUNE50_MODEL_PATH:-${ARTIFACT_ROOT}/v4-prune50-keep128}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/easyep_reproduction}"

V4_PYTHON="${V4_PYTHON:-/opt/sglang-v4/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
RUN_ID="${RUN_ID:-v4flash_easyep_matrix_01}"
REPEATS="${REPEATS:-5}"
WORKERS="${WORKERS:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"

# The official converter retains all MP output state dictionaries in host RAM
# before writing them. These conservative gates can be explicitly overridden.
MIN_FREE_GIB="${MIN_FREE_GIB:-350}"
MIN_AVAILABLE_RAM_GIB="${MIN_AVAILABLE_RAM_GIB:-150}"
ALLOW_LOW_RESOURCES="${ALLOW_LOW_RESOURCES:-0}"

timestamp="$(date -u +%Y%m%d_%H%M%S)"
MASTER_LOG="${MASTER_LOG:-${REPO_ROOT}/logs/v4_full_pipeline_${RUN_ID}_${timestamp}.log}"
CURRENT_STAGE="preflight"

fail() {
  echo "ERROR: $*; stage=${CURRENT_STAGE}" >&2
  echo "Master log: ${MASTER_LOG}" >&2
  exit 2
}

on_error() {
  local status=$?
  echo "PIPELINE FAILED: exit=${status}; stage=${CURRENT_STAGE}" >&2
  echo "Master log: ${MASTER_LOG}" >&2
  exit "${status}"
}
trap on_error ERR

[[ "${GPU_LIST}" == "4,5,6,7" ]] || \
  fail "this verified pipeline is restricted to physical GPUs 4,5,6,7"
[[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || fail "REPEATS must be positive"
[[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || fail "WORKERS must be positive"
[[ "${MIN_FREE_GIB}" =~ ^[0-9]+$ ]] || fail "MIN_FREE_GIB must be an integer"
[[ "${MIN_AVAILABLE_RAM_GIB}" =~ ^[0-9]+$ ]] || \
  fail "MIN_AVAILABLE_RAM_GIB must be an integer"

mkdir -p "${ARTIFACT_ROOT}" "$(dirname "${MASTER_LOG}")" "${RESULTS_ROOT}"
exec > >(tee "${MASTER_LOG}") 2>&1

export FULL_MODEL_PATH V4_INFERENCE_DIR ARTIFACT_ROOT CONVERTED_CKPT_PATH
export PRUNE25_MODEL_PATH PRUNE50_MODEL_PATH RESULTS_ROOT V4_PYTHON GPU_LIST
export RUN_ID REPEATS WORKERS MAX_TOKENS
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

echo "DeepSeek-V4-Flash EASY-EP full pipeline"
echo "Physical GPUs: ${GPU_LIST}"
echo "Full HF model: ${FULL_MODEL_PATH}"
echo "Official inference: ${V4_INFERENCE_DIR}"
echo "MP=4 FP4: ${CONVERTED_CKPT_PATH}"
echo "Prune 25%: ${PRUNE25_MODEL_PATH}"
echo "Prune 50%: ${PRUNE50_MODEL_PATH}"
echo "Results: ${RESULTS_ROOT}/${RUN_ID}"
echo "Downloads/installers: disabled"
echo "Master log: ${MASTER_LOG}"

[[ -x "${V4_PYTHON}" ]] || fail "V4 Python is not executable: ${V4_PYTHON}"
[[ -w "${ARTIFACT_ROOT}" ]] || fail "artifact root is not writable: ${ARTIFACT_ROOT}"
[[ -f "${FULL_MODEL_PATH}/config.json" ]] || \
  fail "full model config is missing: ${FULL_MODEL_PATH}/config.json"
[[ -f "${FULL_MODEL_PATH}/model.safetensors.index.json" ]] || \
  fail "full model index is missing: ${FULL_MODEL_PATH}/model.safetensors.index.json"
[[ -f "${V4_INFERENCE_DIR}/convert.py" ]] || \
  fail "official convert.py is missing: ${V4_INFERENCE_DIR}/convert.py"
[[ -f "${V4_INFERENCE_DIR}/model.py" && -f "${V4_INFERENCE_DIR}/kernel.py" ]] || \
  fail "official model.py/kernel.py are missing from ${V4_INFERENCE_DIR}"
[[ -f "$(dirname "${V4_INFERENCE_DIR}")/encoding/encoding_dsv4.py" ]] || \
  fail "official encoding/encoding_dsv4.py is missing beside ${V4_INFERENCE_DIR}"

"${V4_PYTHON}" -c \
  "import datasets, safetensors, torch, transformers; assert hasattr(torch, 'float4_e2m1fn_x2'); print('V4 conversion/probe dependencies: OK')" || \
  fail "V4 runtime lacks torch FP4, datasets, safetensors, or transformers; no installer was run"

echo "Artifact filesystem:"
df -h "${ARTIFACT_ROOT}"
findmnt -T "${ARTIFACT_ROOT}" -n -o TARGET,SOURCE,FSTYPE,OPTIONS 2>/dev/null || true
echo "Host memory:"
free -h 2>/dev/null || true

validate_mp4() {
  "${V4_PYTHON}" - "${CONVERTED_CKPT_PATH}" <<'PY'
from pathlib import Path
from safetensors import safe_open
import sys

root = Path(sys.argv[1])
for rank in range(4):
    path = root / f"model{rank}-mp4.safetensors"
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty MP=4 shard: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        count = len(list(handle.keys()))
    if count == 0:
        raise SystemExit(f"MP=4 shard has no tensors: {path}")
    print(f"verified {path.name}: tensors={count}, bytes={path.stat().st_size}")
PY
}

shopt -s nullglob
mp4_shards=("${CONVERTED_CKPT_PATH}"/model*-mp4.safetensors)
shopt -u nullglob

if [[ "${#mp4_shards[@]}" -eq 4 ]]; then
  CURRENT_STAGE="verify-existing-mp4"
  validate_mp4
  echo "Reusing complete MP=4 checkpoint."
elif [[ "${#mp4_shards[@]}" -ne 0 ]]; then
  fail "partial MP=4 output found in ${CONVERTED_CKPT_PATH}; use a fresh path or inspect it manually"
else
  CURRENT_STAGE="resource-check-before-conversion"
  available_kib="$(df -Pk "${ARTIFACT_ROOT}" | awk 'NR==2 {print $4}')"
  required_kib="$((MIN_FREE_GIB * 1024 * 1024))"
  available_ram_gib="$(awk '/MemAvailable:/ {print int($2 / 1024 / 1024)}' /proc/meminfo 2>/dev/null || true)"
  available_ram_gib="${available_ram_gib:-0}"
  if [[ "${ALLOW_LOW_RESOURCES}" != "1" ]]; then
    (( available_kib >= required_kib )) || \
      fail "artifact filesystem has less than ${MIN_FREE_GIB} GiB free; set ALLOW_LOW_RESOURCES=1 only after manual review"
    (( available_ram_gib >= MIN_AVAILABLE_RAM_GIB )) || \
      fail "host has less than ${MIN_AVAILABLE_RAM_GIB} GiB MemAvailable; official conversion may OOM"
  fi

  CURRENT_STAGE="convert-hf-to-fp4-mp4"
  mkdir -p "${CONVERTED_CKPT_PATH}"
  conversion_log="${REPO_ROOT}/logs/v4_convert_mp4_fp4_${timestamp}.log"
  echo "Converting existing local HF weights to FP4/MP=4..."
  "${V4_PYTHON}" "${V4_INFERENCE_DIR}/convert.py" \
    --hf-ckpt-path "${FULL_MODEL_PATH}" \
    --save-path "${CONVERTED_CKPT_PATH}" \
    --n-experts 256 \
    --model-parallel 4 \
    --expert-dtype fp4 \
    2>&1 | tee "${conversion_log}"
  validate_mp4
  echo "MP=4 conversion passed: ${conversion_log}"
fi

CURRENT_STAGE="collect-mask-and-materialize"
echo "Collecting/reusing statistics and materializing both pruned checkpoints..."
bash "${SCRIPT_DIR}/prepare_v4_pruned_checkpoints.sh"

CURRENT_STAGE="pruned-checkpoint-acceptance"
echo "Running two load/generate cycles for each pruned checkpoint..."
bash "${SCRIPT_DIR}/validate_v4_pruned_checkpoints_gpus_4_7.sh"

CURRENT_STAGE="evaluation-client-preflight"
supports_math_scoring() {
  "$1" -c \
    "import importlib.util as u; raise SystemExit(0 if (u.find_spec('symeval') or u.find_spec('math_verify')) else 1)" \
    >/dev/null 2>&1
}

if [[ -n "${EVAL_PYTHON}" ]]; then
  command -v "${EVAL_PYTHON}" >/dev/null 2>&1 || \
    [[ -x "${EVAL_PYTHON}" ]] || fail "EVAL_PYTHON is not executable: ${EVAL_PYTHON}"
  supports_math_scoring "${EVAL_PYTHON}" || \
    fail "${EVAL_PYTHON} has neither symeval nor math_verify"
elif supports_math_scoring python3; then
  EVAL_PYTHON="$(command -v python3)"
elif supports_math_scoring "${V4_PYTHON}"; then
  EVAL_PYTHON="${V4_PYTHON}"
else
  fail "no local Python has symeval or math_verify; no installer was run"
fi
export EVAL_PYTHON
echo "Evaluation Python: ${EVAL_PYTHON}"

CURRENT_STAGE="evaluation-dry-run"
echo "Validating the full/prune25/prune50 matrix without launching servers..."
DRY_RUN=1 bash "${SCRIPT_DIR}/run_easyep_reproduction.sh"

if [[ "${DRY_RUN_ONLY}" == "1" ]]; then
  echo "DRY_RUN_ONLY=1: stopping after the validated matrix dry-run."
  echo "Master log: ${MASTER_LOG}"
  exit 0
fi

CURRENT_STAGE="full-evaluation-matrix"
echo "Starting the complete full/prune25/prune50 evaluation matrix..."
DRY_RUN=0 bash "${SCRIPT_DIR}/run_easyep_reproduction.sh"

CURRENT_STAGE="complete"
echo "PASS: full DeepSeek-V4-Flash EASY-EP reproduction pipeline completed."
echo "Report: ${RESULTS_ROOT}/${RUN_ID}/REPORT.md"
echo "Master log: ${MASTER_LOG}"
