#!/usr/bin/env bash
set -euo pipefail

# Required:
#   MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash
# Optional:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PROFILE=smoke PORT=60000

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "MODEL_PATH is required" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TP_SIZE="${TP_SIZE:-8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-60000}"
PROFILE="${PROFILE:-smoke}"
MOE_BACKEND="${MOE_BACKEND:-marlin}"

python3 "${REPO_ROOT}/tools/v4_preflight.py" \
  --model-path "${MODEL_PATH}" \
  --tp "${TP_SIZE}" \
  --backend "${MOE_BACKEND}"

common_args=(
  --trust-remote-code
  --model-path "${MODEL_PATH}"
  --tp "${TP_SIZE}"
  --moe-runner-backend "${MOE_BACKEND}"
  --reasoning-parser deepseek-v4
  --tool-call-parser deepseekv4
  --host "${HOST}"
  --port "${PORT}"
)

case "${PROFILE}" in
  smoke)
    # Eager mode separates weight/backend correctness from CUDA-graph issues.
    # After one request succeeds, restart with PROFILE=verified for evaluation.
    profile_args=(
      --disable-cuda-graph
      --mem-fraction-static 0.85
      # Leave room for the prompt plus a 32K-token completion.
      --context-length 65536
      --max-running-requests 1
    )
    ;;
  verified)
    # Matches SGLang's verified H100/Flash/FP4 high-throughput topology:
    # TP=8 + Marlin, without speculative decoding.
    profile_args=()
    ;;
  *)
    echo "Unsupported PROFILE=${PROFILE}; use smoke or verified" >&2
    exit 2
    ;;
esac

echo "Launching DeepSeek-V4-Flash: profile=${PROFILE}, tp=${TP_SIZE}, backend=${MOE_BACKEND}"
exec sglang serve "${common_args[@]}" "${profile_args[@]}" "$@"
