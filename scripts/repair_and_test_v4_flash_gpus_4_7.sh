#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible entry point for validation plus the four-GPU smoke test.
# Despite the historical filename, this script no longer repairs, installs, or
# downloads anything.  It fails closed when the existing runtime is incomplete.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
VALIDATION_LOG="${LOG_DIR}/v4_validate_and_test_${RUN_ID}.log"
VALIDATION_SUMMARY="${LOG_DIR}/v4_validate_and_test_${RUN_ID}_summary.txt"

mkdir -p "${LOG_DIR}"
echo "Validation and test output will be saved to: ${VALIDATION_LOG}"
echo "Automatic downloads/installations are disabled."

set +e
(
  set -euo pipefail
  bash "${SCRIPT_DIR}/test_v4_flash_gpus_4_7.sh" "$@"
) 2>&1 | tee "${VALIDATION_LOG}"
status=${PIPESTATUS[0]}
set -e

if [[ "${status}" == "0" ]]; then
  result="PASSED"
else
  result="FAILED"
fi
{
  echo "DeepSeek-V4-Flash validation/test wrapper summary"
  echo "================================================"
  echo "Result: ${result}"
  echo "Exit status: ${status}"
  echo "Automatic downloads/installations: disabled"
  echo "Full wrapper log: ${VALIDATION_LOG}"
  echo
  echo "Last 60 log lines"
  echo "-----------------"
  tail -n 60 "${VALIDATION_LOG}" || true
} >"${VALIDATION_SUMMARY}" 2>&1

echo "Validation/test log: ${VALIDATION_LOG}"
echo "Validation/test summary: ${VALIDATION_SUMMARY}"
exit "${status}"
