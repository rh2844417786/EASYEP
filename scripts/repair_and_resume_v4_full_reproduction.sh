#!/usr/bin/env bash
set -Eeuo pipefail

# Repair the one missing official DeepSeek-V4 inference dependency and resume
# the existing four-GPU EASY-EP pipeline. The CUDA, PyTorch, and SGLang
# installations are never changed. The vendored package is built only when its
# import plus a real CUDA Hadamard operation fails.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

V4_PYTHON="${V4_PYTHON:-/opt/sglang-v4/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
FHT_VERSION="${FHT_VERSION:-1.1.0}"
FHT_SOURCE_DIR="${FHT_SOURCE_DIR:-${REPO_ROOT}/third_party/fast-hadamard-transform}"
MAX_JOBS="${MAX_JOBS:-8}"
RUN_ID="${RUN_ID:-v4flash_easyep_matrix_01}"

timestamp="$(date -u +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-${REPO_ROOT}/logs/v4_fht_repair_and_resume_${timestamp}.log}"
SUMMARY_PATH="${SUMMARY_PATH:-${REPO_ROOT}/logs/v4_fht_repair_and_resume_${timestamp}_summary.txt}"
CURRENT_STAGE="preflight"

fail() {
  echo "ERROR: $*; stage=${CURRENT_STAGE}" >&2
  exit 2
}

write_summary() {
  local status=$?
  {
    echo "DeepSeek-V4-Flash FHT repair and resume"
    echo "status=$([[ ${status} -eq 0 ]] && echo PASS || echo FAIL)"
    echo "exit_code=${status}"
    echo "last_stage=${CURRENT_STAGE}"
    echo "python=${V4_PYTHON}"
    echo "cuda_home=${CUDA_HOME}"
    echo "physical_gpus=${GPU_LIST}"
    echo "fast_hadamard_transform_version=${FHT_VERSION}"
    echo "fast_hadamard_transform_source=${FHT_SOURCE_DIR}"
    echo "run_id=${RUN_ID}"
    echo "log=${LOG_PATH}"
  } >"${SUMMARY_PATH}"
  echo "Repair/resume summary: ${SUMMARY_PATH}"
}
trap write_summary EXIT

mkdir -p "$(dirname "${LOG_PATH}")" "$(dirname "${SUMMARY_PATH}")"
exec > >(tee "${LOG_PATH}") 2>&1

[[ "${GPU_LIST}" == "4,5,6,7" ]] || \
  fail "this verified repair/resume path is restricted to physical GPUs 4,5,6,7"
[[ -x "${V4_PYTHON}" ]] || fail "V4 Python is not executable: ${V4_PYTHON}"
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || \
  fail "CUDA compiler is missing: ${CUDA_HOME}/bin/nvcc"
[[ -x "${SCRIPT_DIR}/run_v4_full_reproduction_gpus_4_7.sh" ]] || \
  fail "full reproduction entrypoint is missing"

export CUDA_HOME
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:$(dirname "${V4_PYTHON}"):${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export MAX_JOBS
export V4_PYTHON GPU_LIST RUN_ID

check_fht() {
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${V4_PYTHON}" - <<'PY'
import torch
from fast_hadamard_transform import hadamard_transform

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the selected V4 Python")
if torch.cuda.device_count() != 4:
    raise SystemExit(f"expected 4 visible GPUs, found {torch.cuda.device_count()}")

x = torch.randn(2, 512, device="cuda:0", dtype=torch.bfloat16)
y = hadamard_transform(x, scale=x.size(-1) ** -0.5)
if y.shape != x.shape or y.dtype != x.dtype:
    raise SystemExit(f"unexpected FHT output: shape={tuple(y.shape)}, dtype={y.dtype}")
if not torch.isfinite(y).all().item():
    raise SystemExit("FHT CUDA output contains NaN or Inf")
torch.cuda.synchronize()
print("fast_hadamard_transform CUDA validation: PASS")
print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY
}

check_fht_source() {
  local required
  for required in \
    setup.py \
    LICENSE \
    SHA256SUMS \
    fast_hadamard_transform/__init__.py \
    fast_hadamard_transform/fast_hadamard_transform_interface.py \
    csrc/fast_hadamard_transform.cpp \
    csrc/fast_hadamard_transform_cuda.cu \
    csrc/fast_hadamard_transform.h \
    csrc/fast_hadamard_transform_common.h \
    csrc/fast_hadamard_transform_special.h \
    csrc/static_switch.h; do
    [[ -s "${FHT_SOURCE_DIR}/${required}" ]] || \
      fail "vendored FHT source is missing or empty: ${FHT_SOURCE_DIR}/${required}"
  done
  command -v sha256sum >/dev/null 2>&1 || \
    fail "sha256sum is required to verify the vendored FHT source"
  (cd "${FHT_SOURCE_DIR}" && sha256sum --check SHA256SUMS) || \
    fail "vendored FHT source checksum verification failed"
}

clean_fht_build_artifacts() {
  local source_real build_dir
  source_real="$(cd "${FHT_SOURCE_DIR}" && pwd -P)" || \
    fail "cannot resolve vendored FHT source: ${FHT_SOURCE_DIR}"
  [[ "${source_real}" != "/" ]] || \
    fail "refusing to clean an FHT build directory below the filesystem root"
  build_dir="${source_real}/build"
  if [[ -e "${build_dir}" ]]; then
    echo "Removing stale generated FHT build directory: ${build_dir}"
    rm -rf -- "${build_dir}"
  fi
}

echo "DeepSeek-V4-Flash dependency repair and full-pipeline resume"
echo "Python: ${V4_PYTHON}"
echo "CUDA_HOME: ${CUDA_HOME}"
echo "Physical GPUs: ${GPU_LIST}"
echo "Target dependency: fast-hadamard-transform==${FHT_VERSION}"
echo "Vendored source: ${FHT_SOURCE_DIR}"
echo "CUDA/PyTorch/SGLang reinstall: disabled"
echo "Log: ${LOG_PATH}"

CURRENT_STAGE="check-fast-hadamard-transform"
if check_fht >/dev/null 2>&1; then
  echo "fast_hadamard_transform is already usable; no download or installation is needed."
else
  echo "fast_hadamard_transform is missing or its CUDA extension is unusable."
  command -v uv >/dev/null 2>&1 || \
    fail "uv is required to install the single missing dependency"
  CURRENT_STAGE="prepare-fast-hadamard-transform-source"
  check_fht_source
  clean_fht_build_artifacts
  CURRENT_STAGE="install-fast-hadamard-transform"
  # Version 1.1.0 maps every non-CUDA-11 runtime to a guessed cu122 wheel.
  # CUDA 13 must bypass that lookup and compile the complete vendored source
  # with the already installed nvcc/PyTorch toolchain. The PyPI 1.1.0 sdist is
  # intentionally not used because it omits the csrc build inputs.
  export FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE
  export FAST_HADAMARD_TRANSFORM_SKIP_CUDA_BUILD=FALSE
  echo "Installing only fast-hadamard-transform==${FHT_VERSION} into ${V4_PYTHON}..."
  echo "Network package lookup: disabled; building ${FHT_SOURCE_DIR} with CUDA ${CUDA_HOME}."
  uv pip install \
    --python "${V4_PYTHON}" \
    --offline \
    --no-build-isolation \
    --no-deps \
    --reinstall \
    "${FHT_SOURCE_DIR}"
fi

CURRENT_STAGE="validate-fast-hadamard-transform"
check_fht || fail "fast_hadamard_transform CUDA validation failed after repair"

CURRENT_STAGE="resume-full-reproduction"
echo "Dependency is ready. Resuming the full reproduction pipeline."
echo "Existing complete MP=4 shards will be validated and reused."
bash "${SCRIPT_DIR}/run_v4_full_reproduction_gpus_4_7.sh"

CURRENT_STAGE="complete"
echo "PASS: repair and full reproduction pipeline completed."
