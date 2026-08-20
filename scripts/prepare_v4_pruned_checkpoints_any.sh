#!/usr/bin/env bash
set -euo pipefail

# Generate and materialize one or more DeepSeek-V4-Flash pruning targets from
# one existing EASY-EP calibration probe.  Re-running with new target rates
# reuses TOKEN_STATS and never repeats the GPU calibration unless it is absent.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
V4_PYTHON="${V4_PYTHON:-/opt/sglang-v4/bin/python}"
FULL_MODEL_PATH="${FULL_MODEL_PATH:-/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash}"
TOKEN_STATS="${TOKEN_STATS:-${REPO_ROOT}/expert_statistics/token_information/aime_v4.jsonl}"
AUTO_COLLECT_STATS="${AUTO_COLLECT_STATS:-1}"
MASK_OUTPUT_DIR="${MASK_OUTPUT_DIR:-${REPO_ROOT}/expert_statistics/expert_mask}"
MASK_PREFIX="${MASK_PREFIX:-aime_v4}"
MASK_MANIFEST="${MASK_MANIFEST:-${MASK_OUTPUT_DIR}/${MASK_PREFIX}_mask_manifest.json}"
MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT:-/mnt/docker_data/v4-converted}"
TARGET_EXPERTS="${TARGET_EXPERTS:-}"
PRUNE_PERCENTAGES="${PRUNE_PERCENTAGES:-}"
NUM_SAMPLES="${NUM_SAMPLES:-25}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-longest}"
SEED="${SEED:-42}"
timestamp="$(date -u +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-${REPO_ROOT}/logs/v4_materialize_any_${timestamp}.log}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -x "${V4_PYTHON}" ]] || fail "V4 Python is not executable: ${V4_PYTHON}"
[[ -f "${FULL_MODEL_PATH}/config.json" ]] || \
  fail "full checkpoint config is missing: ${FULL_MODEL_PATH}/config.json"
[[ -f "${FULL_MODEL_PATH}/model.safetensors.index.json" ]] || \
  fail "full checkpoint index is missing: ${FULL_MODEL_PATH}/model.safetensors.index.json"
if [[ -z "${TARGET_EXPERTS}" && -z "${PRUNE_PERCENTAGES}" ]]; then
  fail "set TARGET_EXPERTS and/or PRUNE_PERCENTAGES (comma-separated)"
fi

mkdir -p "$(dirname "${LOG_PATH}")" "${MODEL_OUTPUT_ROOT}"
exec > >(tee "${LOG_PATH}") 2>&1

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

echo "DeepSeek-V4-Flash arbitrary-ratio EASY-EP materialization"
echo "Full checkpoint: ${FULL_MODEL_PATH}"
echo "Calibration statistics: ${TOKEN_STATS}"
echo "Target expert counts: ${TARGET_EXPERTS:-<none>}"
echo "Requested prune percentages: ${PRUNE_PERCENTAGES:-<none>}"
echo "Model output root: ${MODEL_OUTPUT_ROOT}"
echo "Downloads/installers: disabled"
echo "Log: ${LOG_PATH}"

if [[ ! -f "${TOKEN_STATS}" ]]; then
  [[ "${AUTO_COLLECT_STATS}" == "1" ]] || \
    fail "V4 token statistics are missing and AUTO_COLLECT_STATS=0: ${TOKEN_STATS}"
  echo "Calibration statistics are absent; running the one-time GPU collector..."
  TOKEN_STATS="${TOKEN_STATS}" \
    V4_PYTHON="${V4_PYTHON}" \
    FULL_MODEL_PATH="${FULL_MODEL_PATH}" \
    bash "${SCRIPT_DIR}/collect_v4_easyep_statistics_gpus_4_7.sh"
else
  echo "Reusing existing calibration statistics; no model probe is needed."
fi

TARGET_EXPERTS="${TARGET_EXPERTS}" \
PRUNE_PERCENTAGES="${PRUNE_PERCENTAGES}" \
TOKEN_STATS="${TOKEN_STATS}" \
OUTPUT_DIR="${MASK_OUTPUT_DIR}" \
MASK_PREFIX="${MASK_PREFIX}" \
MASK_MANIFEST="${MASK_MANIFEST}" \
NUM_SAMPLES="${NUM_SAMPLES}" \
SAMPLE_STRATEGY="${SAMPLE_STRATEGY}" \
SEED="${SEED}" \
V4_PYTHON="${V4_PYTHON}" \
  bash "${SCRIPT_DIR}/prepare_easyep_masks.sh"

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
    if not mask.is_file():
        raise SystemExit(f"mask listed by manifest is missing: {mask}")
    print(f"{target}\t{mask}\t{model_root / directory}")
PY
}

echo "Preflighting every requested checkpoint without writing weights..."
while IFS=$'\t' read -r target mask_path model_path; do
  [[ -n "${target}" ]] || continue
  echo "Preflight: keep=${target}, output=${model_path}"
  "${V4_PYTHON}" "${REPO_ROOT}/pruning/model_prune_v4.py" \
    --input-dir "${FULL_MODEL_PATH}" \
    --output-dir "${model_path}" \
    --mask-json "${mask_path}" \
    --target-experts "${target}" \
    --dry-run
done < <(read_plans)

echo "Applying/upgrading the version-checked SGLang EASY-EP mask-routing patch..."
"${V4_PYTHON}" "${SCRIPT_DIR}/patch_sglang_v4_heterogeneous_experts.py" --apply

while IFS=$'\t' read -r target mask_path model_path; do
  [[ -n "${target}" ]] || continue
  echo "Materializing: keep=${target}, output=${model_path}"
  "${V4_PYTHON}" "${REPO_ROOT}/pruning/model_prune_v4.py" \
    --input-dir "${FULL_MODEL_PATH}" \
    --output-dir "${model_path}" \
    --mask-json "${mask_path}" \
    --target-experts "${target}"
  "${V4_PYTHON}" "${REPO_ROOT}/pruning/model_prune_v4.py" \
    --input-dir "${FULL_MODEL_PATH}" \
    --output-dir "${model_path}" \
    --mask-json "${mask_path}" \
    --target-experts "${target}" \
    --verify-only
done < <(read_plans)

"${V4_PYTHON}" "${SCRIPT_DIR}/patch_sglang_v4_heterogeneous_experts.py" --check

echo "PASS: every requested checkpoint was materialized and structurally verified."
echo "Mask manifest: ${MASK_MANIFEST}"
echo "Validation: MASK_MANIFEST=${MASK_MANIFEST} MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT} bash scripts/validate_v4_pruned_checkpoints_any_gpus_4_7.sh"
echo "Full log: ${LOG_PATH}"
