#!/usr/bin/env bash
set -euo pipefail

# Collect the paper-defined EASY-EP calibration statistics for DeepSeek-V4-Flash
# on physical GPUs 4..7. This script is offline-only: it never installs,
# downloads, or converts model weights.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
V4_PYTHON="${V4_PYTHON:-/opt/sglang-v4/bin/python}"
FULL_MODEL_PATH="${FULL_MODEL_PATH:-/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash}"
V4_INFERENCE_DIR="${V4_INFERENCE_DIR:-}"
CONVERTED_CKPT_PATH="${CONVERTED_CKPT_PATH:-/mnt/docker_data/v4-converted}"
PROBE_CONFIG="${PROBE_CONFIG:-${REPO_ROOT}/configs/config_v4_flash.json}"
SOURCE_CALIBRATION="${SOURCE_CALIBRATION:-${REPO_ROOT}/dataset/aime23_full}"
V4_CALIBRATION="${V4_CALIBRATION:-${REPO_ROOT}/dataset/aime23_full_v4}"
TOKEN_STATS="${TOKEN_STATS:-${REPO_ROOT}/expert_statistics/token_information/aime_v4.jsonl}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
CALIBRATION_LIMIT="${CALIBRATION_LIMIT:-25}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-13000}"
TRACE_FIRST_SAMPLE="${TRACE_FIRST_SAMPLE:-1}"
SYNC_DEBUG="${SYNC_DEBUG:-0}"
timestamp="$(date -u +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-${REPO_ROOT}/logs/v4_easyep_statistics_${timestamp}.log}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ "${CALIBRATION_LIMIT}" =~ ^[1-9][0-9]*$ ]] || \
  fail "CALIBRATION_LIMIT must be a positive integer"
[[ "${MAX_INPUT_TOKENS}" =~ ^[1-9][0-9]*$ ]] || \
  fail "MAX_INPUT_TOKENS must be a positive integer"

IFS=',' read -r -a physical_gpus <<<"${GPU_LIST}"
nproc="${#physical_gpus[@]}"
[[ "${nproc}" -eq 4 ]] || \
  fail "this verified entrypoint requires exactly four physical GPUs; got ${GPU_LIST}"
[[ "${GPU_LIST}" == "4,5,6,7" ]] || \
  fail "this entrypoint is intentionally restricted to physical GPUs 4,5,6,7"

[[ -x "${V4_PYTHON}" ]] || fail "V4 Python is not executable: ${V4_PYTHON}"
[[ -d "${SOURCE_CALIBRATION}" ]] || \
  fail "source calibration dataset is missing: ${SOURCE_CALIBRATION}"
[[ -f "${FULL_MODEL_PATH}/config.json" ]] || \
  fail "local V4 checkpoint is missing config.json: ${FULL_MODEL_PATH}"
[[ -f "${PROBE_CONFIG}" ]] || fail "probe config is missing: ${PROBE_CONFIG}"

if [[ -z "${V4_INFERENCE_DIR}" ]]; then
  for candidate in \
    "${FULL_MODEL_PATH}/inference" \
    "${REPO_ROOT}/DeepSeek-V4-Flash/inference" \
    "/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-code/inference"; do
    if [[ -f "${candidate}/model.py" && -f "${candidate}/kernel.py" ]]; then
      V4_INFERENCE_DIR="${candidate}"
      break
    fi
  done
fi
[[ -n "${V4_INFERENCE_DIR}" ]] || fail \
  "official V4 inference code was not found; export V4_INFERENCE_DIR=/path/to/DeepSeek-V4-Flash/inference"
[[ -f "${V4_INFERENCE_DIR}/model.py" && -f "${V4_INFERENCE_DIR}/kernel.py" ]] || \
  fail "V4_INFERENCE_DIR must contain official model.py and kernel.py: ${V4_INFERENCE_DIR}"
[[ -f "$(dirname "${V4_INFERENCE_DIR}")/encoding/encoding_dsv4.py" ]] || \
  fail "official encoding/encoding_dsv4.py is missing beside ${V4_INFERENCE_DIR}"

for ((rank = 0; rank < nproc; rank++)); do
  [[ -f "${CONVERTED_CKPT_PATH}/model${rank}-mp${nproc}.safetensors" ]] || \
    fail "missing MP=${nproc} converted shard: ${CONVERTED_CKPT_PATH}/model${rank}-mp${nproc}.safetensors"
done

mkdir -p "$(dirname "${LOG_PATH}")" "$(dirname "${TOKEN_STATS}")"
exec > >(tee "${LOG_PATH}") 2>&1

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"

echo "DeepSeek-V4-Flash EASY-EP statistic collection"
echo "Physical GPUs: ${GPU_LIST} (torchrun ranks 0..$((nproc - 1)))"
echo "Official inference: ${V4_INFERENCE_DIR}"
echo "Converted MP=${nproc} checkpoint: ${CONVERTED_CKPT_PATH}"
echo "Calibration source: ${SOURCE_CALIBRATION}"
echo "V4 calibration: ${V4_CALIBRATION}"
echo "Statistics: ${TOKEN_STATS}"
echo "Samples: ${CALIBRATION_LIMIT}"
echo "Downloads/installers/conversion: disabled"
echo "Log: ${LOG_PATH}"

"${V4_PYTHON}" -c \
  "import datasets, safetensors, torch, transformers; from fast_hadamard_transform import hadamard_transform; x=torch.randn(2,512,device='cuda:0',dtype=torch.bfloat16); y=hadamard_transform(x,scale=x.size(-1)**-0.5); assert y.shape==x.shape and torch.isfinite(y).all(); torch.cuda.synchronize(); print('runtime dependencies: OK')" || \
  fail "V4 runtime is missing a working torch/datasets/safetensors/transformers/fast_hadamard_transform dependency; run scripts/repair_and_resume_v4_full_reproduction.sh"

if [[ ! -d "${V4_CALIBRATION}" ]]; then
  echo "Retokenizing the local AIME23 calibration set with the local V4 tokenizer..."
  "${V4_PYTHON}" "${REPO_ROOT}/pruning/prepare_v4_calibration.py" \
    --source "${SOURCE_CALIBRATION}" \
    --output "${V4_CALIBRATION}" \
    --model-path "${FULL_MODEL_PATH}" \
    --inference-dir "${V4_INFERENCE_DIR}"
else
  echo "V4 calibration already exists; reusing ${V4_CALIBRATION}"
fi

probe_args=(
  --nproc-per-node "${nproc}"
  "${REPO_ROOT}/pruning/inf_v4.py"
  --inference-dir "${V4_INFERENCE_DIR}"
  --ckpt-path "${CONVERTED_CKPT_PATH}"
  --config "${PROBE_CONFIG}"
  --input-file "${V4_CALIBRATION}"
  --output "${TOKEN_STATS}"
  --limit "${CALIBRATION_LIMIT}"
  --max-input-tokens "${MAX_INPUT_TOKENS}"
  --resume
)
[[ "${TRACE_FIRST_SAMPLE}" == "1" ]] && probe_args+=(--trace-first-sample)
[[ "${SYNC_DEBUG}" == "1" ]] && probe_args+=(--sync-debug)

echo "Collecting/resuming paper-defined statistics..."
"${V4_PYTHON}" -m torch.distributed.run "${probe_args[@]}"

record_count="$("${V4_PYTHON}" -c \
  'import pathlib,sys; p=pathlib.Path(sys.argv[1]); print(sum(1 for x in p.open(encoding="utf-8") if x.strip()))' \
  "${TOKEN_STATS}")"
[[ "${record_count}" -eq "${CALIBRATION_LIMIT}" ]] || \
  fail "statistics record count is ${record_count}, expected ${CALIBRATION_LIMIT}"

echo "PASS: ${record_count} V4 EASY-EP calibration records are ready."
echo "Next: bash scripts/prepare_v4_pruned_checkpoints.sh"
echo "Statistics log: ${LOG_PATH}"
