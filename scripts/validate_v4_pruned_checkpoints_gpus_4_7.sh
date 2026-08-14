#!/usr/bin/env bash
set -euo pipefail

# Structural verification plus load/generate/stop/reload acceptance for both
# mask-routed DeepSeek-V4-Flash checkpoints. No download or install occurs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
V4_PYTHON="${V4_PYTHON:-/opt/sglang-v4/bin/python}"
FULL_MODEL_PATH="${FULL_MODEL_PATH:-/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash}"
MASK25="${MASK25:-${REPO_ROOT}/expert_statistics/expert_mask/aime_v4_prune25_keep192.json}"
MASK50="${MASK50:-${REPO_ROOT}/expert_statistics/expert_mask/aime_v4_prune50_keep128.json}"
PRUNE25_MODEL_PATH="${PRUNE25_MODEL_PATH:-${REPO_ROOT}/models/v4-prune25-keep192}"
PRUNE50_MODEL_PATH="${PRUNE50_MODEL_PATH:-${REPO_ROOT}/models/v4-prune50-keep128}"
RELOAD_PASSES="${RELOAD_PASSES:-2}"
PORT="${PORT:-60000}"
timestamp="$(date -u +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-${REPO_ROOT}/logs/v4_pruned_acceptance_${timestamp}.log}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ "${RELOAD_PASSES}" =~ ^[1-9][0-9]*$ ]] || \
  fail "RELOAD_PASSES must be a positive integer"
[[ -x "${V4_PYTHON}" ]] || fail "V4 Python is not executable: ${V4_PYTHON}"
mkdir -p "$(dirname "${LOG_PATH}")"
exec > >(tee "${LOG_PATH}") 2>&1

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

echo "DeepSeek-V4-Flash pruned-checkpoint acceptance"
echo "Hash layers 0..2: 256 experts, unchanged"
echo "Dynamic layers 3..42: 192 or 128 experts"
echo "Router weight/bias: 256 rows, masked before TopK"
echo "Load/generate passes per checkpoint: ${RELOAD_PASSES}"
echo "Physical GPUs: 4,5,6,7"
echo "Downloads/installers: disabled"
echo "Log: ${LOG_PATH}"

"${V4_PYTHON}" "${SCRIPT_DIR}/patch_sglang_v4_heterogeneous_experts.py" --check || \
  fail "the SGLang 0.5.16 EASY-EP mask-routing patch is not active"

echo "Verifying 25%-dynamic-layer checkpoint structure..."
"${V4_PYTHON}" "${REPO_ROOT}/pruning/model_prune_v4.py" \
  --input-dir "${FULL_MODEL_PATH}" \
  --output-dir "${PRUNE25_MODEL_PATH}" \
  --mask-json "${MASK25}" \
  --target-experts 192 \
  --verify-only

echo "Verifying 50%-dynamic-layer checkpoint structure..."
"${V4_PYTHON}" "${REPO_ROOT}/pruning/model_prune_v4.py" \
  --input-dir "${FULL_MODEL_PATH}" \
  --output-dir "${PRUNE50_MODEL_PATH}" \
  --mask-json "${MASK50}" \
  --target-experts 128 \
  --verify-only

run_reload_test() {
  local variant="$1"
  local model_path="$2"
  local pass
  for ((pass = 1; pass <= RELOAD_PASSES; pass++)); do
    echo "${variant}: load/generate pass ${pass}/${RELOAD_PASSES}"
    MODEL_PATH="${model_path}" \
      V4_PYTHON="${V4_PYTHON}" \
      PORT="${PORT}" \
      bash "${SCRIPT_DIR}/test_v4_flash_gpus_4_7.sh"
  done
}

run_reload_test "prune25" "${PRUNE25_MODEL_PATH}"
run_reload_test "prune50" "${PRUNE50_MODEL_PATH}"

echo "PASS: both pruned checkpoints passed structural verification and ${RELOAD_PASSES} load/generate cycle(s)."
echo "Acceptance log: ${LOG_PATH}"
echo "Next: RUN_ID=v4flash_easyep_matrix_01 bash scripts/run_easyep_reproduction.sh"
