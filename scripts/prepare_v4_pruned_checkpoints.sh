#!/usr/bin/env bash
set -euo pipefail

# Materialize two DeepSeek-V4-Flash checkpoints with paper-aligned masks.
# No dependency, model, or dataset download is performed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
V4_PYTHON="${V4_PYTHON:-/opt/sglang-v4/bin/python}"
FULL_MODEL_PATH="${FULL_MODEL_PATH:-/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash}"
TOKEN_STATS="${TOKEN_STATS:-${REPO_ROOT}/expert_statistics/token_information/aime_v4.jsonl}"
MASK25="${MASK25:-${REPO_ROOT}/expert_statistics/expert_mask/aime_v4_prune25_keep192.json}"
MASK50="${MASK50:-${REPO_ROOT}/expert_statistics/expert_mask/aime_v4_prune50_keep128.json}"
PRUNE25_MODEL_PATH="${PRUNE25_MODEL_PATH:-${REPO_ROOT}/models/v4-prune25-keep192}"
PRUNE50_MODEL_PATH="${PRUNE50_MODEL_PATH:-${REPO_ROOT}/models/v4-prune50-keep128}"
timestamp="$(date -u +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-${REPO_ROOT}/logs/v4_materialize_pruned_${timestamp}.log}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

mkdir -p "$(dirname "${LOG_PATH}")"
exec > >(tee "${LOG_PATH}") 2>&1

echo "DeepSeek-V4-Flash physical pruning"
echo "Hash layers 0..2: preserve 256/256 experts"
echo "Dynamic layers 3..42: materialize 192/256 and 128/256 experts"
echo "Router weight/bias: preserve all 256 rows; apply EASY-EP mask at runtime"
echo "Input: ${FULL_MODEL_PATH}"
echo "25% output: ${PRUNE25_MODEL_PATH}"
echo "50% output: ${PRUNE50_MODEL_PATH}"
echo "Log: ${LOG_PATH}"
echo "Downloads/installers: disabled"

[[ -x "${V4_PYTHON}" ]] || fail "V4 Python is not executable: ${V4_PYTHON}"
[[ -f "${FULL_MODEL_PATH}/config.json" ]] || \
  fail "full checkpoint config is missing: ${FULL_MODEL_PATH}/config.json"
[[ -f "${FULL_MODEL_PATH}/model.safetensors.index.json" ]] || \
  fail "full checkpoint index is missing: ${FULL_MODEL_PATH}/model.safetensors.index.json"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

"${V4_PYTHON}" -c \
  "import torch, safetensors; print('torch:', torch.__version__); print('safetensors:', safetensors.__version__)" || \
  fail "the active V4 runtime needs torch and safetensors; no installer was run"

if [[ ! -f "${MASK25}" || ! -f "${MASK50}" ]]; then
  [[ -f "${TOKEN_STATS}" ]] || \
    fail "V4 token statistics are missing: ${TOKEN_STATS}; complete calibration/probe first"
  TOKEN_STATS="${TOKEN_STATS}" V4_PYTHON="${V4_PYTHON}" \
    bash "${SCRIPT_DIR}/prepare_easyep_masks_25_50.sh"
fi

echo "Preflighting both checkpoint plans without writing weights..."
"${V4_PYTHON}" "${REPO_ROOT}/pruning/model_prune_v4.py" \
  --input-dir "${FULL_MODEL_PATH}" \
  --output-dir "${PRUNE25_MODEL_PATH}" \
  --mask-json "${MASK25}" \
  --target-experts 192 \
  --dry-run
"${V4_PYTHON}" "${REPO_ROOT}/pruning/model_prune_v4.py" \
  --input-dir "${FULL_MODEL_PATH}" \
  --output-dir "${PRUNE50_MODEL_PATH}" \
  --mask-json "${MASK50}" \
  --target-experts 128 \
  --dry-run

echo "Applying the version-checked SGLang EASY-EP mask-routing patch..."
"${V4_PYTHON}" "${SCRIPT_DIR}/patch_sglang_v4_heterogeneous_experts.py" --apply

echo "Materializing the 25%-dynamic-layer checkpoint..."
"${V4_PYTHON}" "${REPO_ROOT}/pruning/model_prune_v4.py" \
  --input-dir "${FULL_MODEL_PATH}" \
  --output-dir "${PRUNE25_MODEL_PATH}" \
  --mask-json "${MASK25}" \
  --target-experts 192

echo "Materializing the 50%-dynamic-layer checkpoint..."
"${V4_PYTHON}" "${REPO_ROOT}/pruning/model_prune_v4.py" \
  --input-dir "${FULL_MODEL_PATH}" \
  --output-dir "${PRUNE50_MODEL_PATH}" \
  --mask-json "${MASK50}" \
  --target-experts 128

"${V4_PYTHON}" "${SCRIPT_DIR}/patch_sglang_v4_heterogeneous_experts.py" --check

echo "Both pruned checkpoints were materialized and structurally verified."
echo "Next: RUN_ID=v4flash_easyep_matrix_01 bash scripts/run_easyep_reproduction.sh"
echo "Full log: ${LOG_PATH}"
