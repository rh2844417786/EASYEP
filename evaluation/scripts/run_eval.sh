#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_URL="${BASE_URL:-http://127.0.0.1:60000/v1}"
TARGET_PATH="${TARGET_PATH:-${REPO_ROOT}/evaluation/outputs}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
REPEATS="${REPEATS:-5}"
WORKERS="${WORKERS:-8}"
MODEL_NAME="${MODEL_NAME:-}"

model_args=()
if [[ -n "${MODEL_NAME}" ]]; then
  model_args+=(--model "${MODEL_NAME}")
fi

for dataset in AIME24 hmmt_feb_2025 AIME25; do
  python3 "${REPO_ROOT}/evaluation/run_sglang.py" \
    --data-name "${dataset}" \
    --target-path "${TARGET_PATH}" \
    --base-url "${BASE_URL}" \
    --max-tokens "${MAX_TOKENS}" \
    --repeats "${REPEATS}" \
    --workers "${WORKERS}" \
    --thinking \
    "${model_args[@]}"
done
