#!/usr/bin/env bash
set -euo pipefail

# Structurally verify and load/generate/reload every checkpoint listed in a
# generic EASY-EP mask manifest. No download or install is performed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
V4_PYTHON="${V4_PYTHON:-/opt/sglang-v4/bin/python}"
FULL_MODEL_PATH="${FULL_MODEL_PATH:-/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash}"
MASK_OUTPUT_DIR="${MASK_OUTPUT_DIR:-${REPO_ROOT}/expert_statistics/expert_mask}"
MASK_PREFIX="${MASK_PREFIX:-aime_v4}"
MASK_MANIFEST="${MASK_MANIFEST:-${MASK_OUTPUT_DIR}/${MASK_PREFIX}_mask_manifest.json}"
MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT:-/mnt/docker_data/v4-converted}"
RELOAD_PASSES="${RELOAD_PASSES:-2}"
PORT="${PORT:-60000}"
timestamp="$(date -u +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-${REPO_ROOT}/logs/v4_pruned_any_acceptance_${timestamp}.log}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ "${RELOAD_PASSES}" =~ ^[1-9][0-9]*$ ]] || \
  fail "RELOAD_PASSES must be a positive integer"
[[ -x "${V4_PYTHON}" ]] || fail "V4 Python is not executable: ${V4_PYTHON}"
[[ -f "${MASK_MANIFEST}" ]] || fail "mask manifest is missing: ${MASK_MANIFEST}"
mkdir -p "$(dirname "${LOG_PATH}")"
exec > >(tee "${LOG_PATH}") 2>&1

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

echo "DeepSeek-V4-Flash arbitrary-ratio checkpoint acceptance"
echo "Mask manifest: ${MASK_MANIFEST}"
echo "Model output root: ${MODEL_OUTPUT_ROOT}"
echo "Load/generate passes per checkpoint: ${RELOAD_PASSES}"
echo "Physical GPUs: 4,5,6,7"
echo "Downloads/installers: disabled"
echo "Log: ${LOG_PATH}"

"${V4_PYTHON}" "${SCRIPT_DIR}/patch_sglang_v4_heterogeneous_experts.py" --check || \
  fail "the current SGLang EASY-EP arbitrary-ratio patch is not active"

read_plans() {
  "${V4_PYTHON}" - "${MASK_MANIFEST}" "${MODEL_OUTPUT_ROOT}" <<'PY'
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
model_root = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("format_version") != 2:
    raise SystemExit(
        f"unsupported mask manifest format in {manifest_path}; "
        "rerun scripts/prepare_easyep_masks.sh"
    )
records = manifest.get("masks")
if not isinstance(records, list) or not records:
    raise SystemExit(f"mask manifest has no records: {manifest_path}")
for record in records:
    target = int(record["dynamic_layer_experts"])
    mask = Path(record["path"])
    directory = str(record["model_directory_name"])
    if not directory or directory in {".", ".."} or Path(directory).name != directory:
        raise SystemExit(f"unsafe model directory in mask manifest: {directory!r}")
    model = model_root / directory
    print(f"{target}\t{mask}\t{model}")
PY
}

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

while IFS=$'\t' read -r target mask_path model_path; do
  [[ -n "${target}" ]] || continue
  echo "Verifying keep=${target}: ${model_path}"
  "${V4_PYTHON}" "${REPO_ROOT}/pruning/model_prune_v4.py" \
    --input-dir "${FULL_MODEL_PATH}" \
    --output-dir "${model_path}" \
    --mask-json "${mask_path}" \
    --target-experts "${target}" \
    --verify-only
  run_reload_test "keep${target}" "${model_path}"
done < <(read_plans)

echo "PASS: every manifest checkpoint passed structural and load/generate/reload acceptance."
echo "Acceptance log: ${LOG_PATH}"
