#!/usr/bin/env bash
set -euo pipefail

# Explicit, one-time preparation of the lightweight math-scoring client.
# Nothing is downloaded unless ALLOW_EVAL_DEP_INSTALL=1 is supplied.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
V4_PYTHON="${V4_PYTHON:-/opt/sglang-v4/bin/python}"
EVAL_ENV_DIR="${EVAL_ENV_DIR:-/opt/easyep-eval}"
EVAL_PYTHON="${EVAL_PYTHON:-${EVAL_ENV_DIR}/bin/python}"
ALLOW_EVAL_DEP_INSTALL="${ALLOW_EVAL_DEP_INSTALL:-0}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

supports_math_scoring() {
  "$1" -c \
    "import importlib.util as u, sys; raise SystemExit(0 if sys.version_info >= (3, 10) and (u.find_spec('symeval') or u.find_spec('math_verify')) else 1)" \
    >/dev/null 2>&1
}

[[ "${ALLOW_EVAL_DEP_INSTALL}" =~ ^[01]$ ]] || \
  fail "ALLOW_EVAL_DEP_INSTALL must be 0 or 1"
[[ -x "${V4_PYTHON}" ]] || fail "V4 Python is not executable: ${V4_PYTHON}"

if [[ -x "${EVAL_PYTHON}" ]] && supports_math_scoring "${EVAL_PYTHON}"; then
  echo "Evaluation runtime already ready: ${EVAL_PYTHON}"
  "${EVAL_PYTHON}" -c \
    "from evaluation.evaluator.MATH_evaluator_list import MATHEvaluator; print('Evaluator:', MATHEvaluator().backend)" \
    2>/dev/null || \
    PYTHONPATH="${REPO_ROOT}" "${EVAL_PYTHON}" -c \
      "from evaluation.evaluator.MATH_evaluator_list import MATHEvaluator; print('Evaluator:', MATHEvaluator().backend)"
  exit 0
fi

[[ "${ALLOW_EVAL_DEP_INSTALL}" == "1" ]] || \
  fail "evaluation runtime is missing; no installer was run. Re-run with ALLOW_EVAL_DEP_INSTALL=1"
command -v uv >/dev/null 2>&1 || fail "uv is required for the explicit evaluation install"

mkdir -p "${EVAL_ENV_DIR}" "${LOG_DIR}"
if [[ ! -x "${EVAL_PYTHON}" ]]; then
  uv venv --python "${V4_PYTHON}" "${EVAL_ENV_DIR}"
fi

timestamp="$(date -u +%Y%m%d_%H%M%S)"
log_file="${LOG_DIR}/easyep_eval_runtime_${timestamp}.log"
echo "Installing only requirements-eval.txt into ${EVAL_ENV_DIR}"
echo "Install log: ${log_file}"
set -o pipefail
uv pip install \
  --python "${EVAL_PYTHON}" \
  -r "${REPO_ROOT}/requirements-eval.txt" \
  2>&1 | tee "${log_file}"

supports_math_scoring "${EVAL_PYTHON}" || \
  fail "installation completed but no math-scoring backend is importable; see ${log_file}"
PYTHONPATH="${REPO_ROOT}" "${EVAL_PYTHON}" -c \
  "from evaluation.evaluator.MATH_evaluator_list import MATHEvaluator; print('Evaluator:', MATHEvaluator().backend)"
echo "Evaluation runtime ready: ${EVAL_PYTHON}"
