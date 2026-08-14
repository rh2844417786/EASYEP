#!/usr/bin/env bash
set -euo pipefail

# One-command repair and smoke test for the current Docker container.
# The installer is sourced so CUDA_HOME and DG_JIT_NVCC_COMPILER are inherited
# by the SGLang launcher in the same process tree.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
REPAIR_LOG="${LOG_DIR}/v4_repair_and_test_${RUN_ID}.log"

mkdir -p "${LOG_DIR}"
echo "Repair/install output and the test launcher output will be saved to: ${REPAIR_LOG}"

set +e
(
  set -euo pipefail
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/install_v4_cuda_toolchain.sh"
  echo "Starting the four-GPU test with NVCC: ${DG_JIT_NVCC_COMPILER}"
  bash "${SCRIPT_DIR}/test_v4_flash_gpus_4_7.sh" "$@"
) 2>&1 | tee "${REPAIR_LOG}"
status=${PIPESTATUS[0]}
set -e

echo "Repair/test log: ${REPAIR_LOG}"
exit "${status}"
