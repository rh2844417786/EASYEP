#!/usr/bin/env bash
set -euo pipefail

# Generate one or more V4 EASY-EP masks from an existing calibration probe.
# This is CPU-only mask generation. It never launches the model, downloads
# data, or materializes checkpoint weights.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
V4_PYTHON="${V4_PYTHON:-/opt/sglang-v4/bin/python}"
TOKEN_STATS="${TOKEN_STATS:-${REPO_ROOT}/expert_statistics/token_information/aime_v4.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/expert_statistics/expert_mask}"
MASK_PREFIX="${MASK_PREFIX:-aime_v4}"
TARGET_EXPERTS="${TARGET_EXPERTS:-}"
PRUNE_PERCENTAGES="${PRUNE_PERCENTAGES:-}"
NUM_EXPERTS="${NUM_EXPERTS:-256}"
NUM_LAYERS="${NUM_LAYERS:-43}"
HASH_LAYERS="${HASH_LAYERS:-3}"
NUM_SAMPLES="${NUM_SAMPLES:-25}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-longest}"
SEED="${SEED:-42}"
SCORES_FILE="${SCORES_FILE:-${OUTPUT_DIR}/${MASK_PREFIX}_scores.pt}"
MASK_MANIFEST="${MASK_MANIFEST:-${OUTPUT_DIR}/${MASK_PREFIX}_mask_manifest.json}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -x "${V4_PYTHON}" ]] || fail "V4 Python is not executable: ${V4_PYTHON}"
[[ -f "${TOKEN_STATS}" ]] || fail "calibration statistics are missing: ${TOKEN_STATS}"
[[ "${NUM_EXPERTS}" == "256" ]] || \
  fail "DeepSeek-V4-Flash mask generation requires NUM_EXPERTS=256"
[[ "${NUM_LAYERS}" == "43" && "${HASH_LAYERS}" == "3" ]] || \
  fail "DeepSeek-V4-Flash requires 43 layers with the first 3 hash layers preserved"
if [[ -z "${TARGET_EXPERTS}" && -z "${PRUNE_PERCENTAGES}" ]]; then
  fail "set TARGET_EXPERTS and/or PRUNE_PERCENTAGES (comma-separated)"
fi

generator_args=(
  --input-file "${TOKEN_STATS}"
  --output-dir "${OUTPUT_DIR}"
  --scores-file "${SCORES_FILE}"
  --manifest "${MASK_MANIFEST}"
  --mask-prefix "${MASK_PREFIX}"
  --num-experts "${NUM_EXPERTS}"
  --num-layers "${NUM_LAYERS}"
  --hash-layers "${HASH_LAYERS}"
  --num-samples "${NUM_SAMPLES}"
  --sample-strategy "${SAMPLE_STRATEGY}"
  --seed "${SEED}"
)

if [[ -n "${TARGET_EXPERTS}" ]]; then
  IFS=',' read -r -a target_values <<<"${TARGET_EXPERTS}"
  for target in "${target_values[@]}"; do
    [[ "${target}" =~ ^[1-9][0-9]*$ ]] || \
      fail "TARGET_EXPERTS contains a non-positive integer: ${target}"
    generator_args+=(--target-experts "${target}")
  done
fi

if [[ -n "${PRUNE_PERCENTAGES}" ]]; then
  IFS=',' read -r -a prune_values <<<"${PRUNE_PERCENTAGES}"
  for percent in "${prune_values[@]}"; do
    [[ -n "${percent}" ]] || fail "PRUNE_PERCENTAGES contains an empty value"
    generator_args+=(--prune-percent "${percent}")
  done
fi

echo "Generating V4 EASY-EP masks from one calibration probe"
echo "Token statistics: ${TOKEN_STATS}"
echo "Target expert counts: ${TARGET_EXPERTS:-<none>}"
echo "Requested prune percentages: ${PRUNE_PERCENTAGES:-<none>}"
echo "Hash layers 0..$((HASH_LAYERS - 1)): preserve all ${NUM_EXPERTS} experts"
echo "Downloads/model execution/checkpoint writes: disabled"

"${V4_PYTHON}" "${REPO_ROOT}/pruning/generate_v4_masks.py" "${generator_args[@]}"

echo "PASS: masks were generated from the existing calibration statistics."
echo "Manifest: ${MASK_MANIFEST}"
