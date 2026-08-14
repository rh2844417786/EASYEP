#!/usr/bin/env bash
set -euo pipefail

# Install a DeepGEMM-compatible CUDA compiler into the current Docker image.
# This intentionally installs toolkit/compiler packages only; it never installs
# or replaces the host NVIDIA driver.

# SGLang 0.5.16 defaults to a CUDA 13 runtime.  Keep the toolkit version
# overridable for older images, but install the matching CUDA 13 compiler when
# the current container does not already provide a compatible NVCC.
CUDA_TOOLKIT_DOT="${CUDA_TOOLKIT_DOT:-13.0}"
CUDA_TOOLKIT_SERIES="${CUDA_TOOLKIT_SERIES:-${CUDA_TOOLKIT_DOT//./-}}"
CUDA_TOOLKIT_ROOT="${CUDA_TOOLKIT_ROOT:-/usr/local/cuda-${CUDA_TOOLKIT_DOT}}"
CUDA_APT_PACKAGES="${CUDA_APT_PACKAGES:-cuda-compiler-${CUDA_TOOLKIT_SERIES} cuda-cudart-dev-${CUDA_TOOLKIT_SERIES} cuda-cccl-${CUDA_TOOLKIT_SERIES}}"
CUDA_REPO_BASE_URL="${CUDA_REPO_BASE_URL:-}"
MIN_NVCC_MAJOR=12
MIN_NVCC_MINOR=3

fail() {
  echo "ERROR: $*" >&2
  return 1
}

nvcc_release() {
  local compiler="$1"
  "${compiler}" --version 2>/dev/null | sed -n \
    's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1.\2/p' | head -n 1
}

nvcc_is_compatible() {
  local release="$1"
  local major minor
  [[ "${release}" =~ ^([0-9]+)\.([0-9]+)$ ]] || return 1
  major="${BASH_REMATCH[1]}"
  minor="${BASH_REMATCH[2]}"
  (( major > MIN_NVCC_MAJOR || \
    (major == MIN_NVCC_MAJOR && minor >= MIN_NVCC_MINOR) ))
}

nvcc_is_newer() {
  local candidate_release="$1"
  local current_release="$2"
  local candidate_major candidate_minor current_major current_minor
  [[ "${candidate_release}" =~ ^([0-9]+)\.([0-9]+)$ ]] || return 1
  candidate_major="${BASH_REMATCH[1]}"
  candidate_minor="${BASH_REMATCH[2]}"
  [[ "${current_release}" =~ ^([0-9]+)\.([0-9]+)$ ]] || return 0
  current_major="${BASH_REMATCH[1]}"
  current_minor="${BASH_REMATCH[2]}"
  (( candidate_major > current_major || \
    (candidate_major == current_major && candidate_minor > current_minor) ))
}

activate_compiler() {
  local compiler="$1"
  local release
  [[ -x "${compiler}" ]] || return 1
  compiler="$(readlink -f "${compiler}" 2>/dev/null || printf '%s' "${compiler}")"
  release="$(nvcc_release "${compiler}" || true)"
  nvcc_is_compatible "${release}" || return 1

  export DG_JIT_NVCC_COMPILER="${compiler}"
  export CUDA_HOME="$(cd "$(dirname "${compiler}")/.." && pwd)"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  echo "Activated NVCC ${release}: ${DG_JIT_NVCC_COMPILER}"
  echo "CUDA_HOME=${CUDA_HOME}"
}

report_cuda_context() {
  local driver_cuda="" python_bin="" runtime_cuda=""
  local candidate

  if command -v nvidia-smi >/dev/null 2>&1; then
    driver_cuda="$(nvidia-smi 2>/dev/null | sed -n \
      's/.*CUDA Version: \([^ ]*\).*/\1/p' | head -n 1 || true)"
    [[ -n "${driver_cuda}" ]] && \
      echo "NVIDIA driver maximum supported CUDA: ${driver_cuda} (this is not the NVCC version)"
  fi

  for candidate in "${V4_PYTHON:-}" "/opt/sglang-v4/bin/python"; do
    [[ -n "${candidate}" && -x "${candidate}" ]] || continue
    python_bin="${candidate}"
    break
  done
  if [[ -n "${python_bin}" ]] && \
    runtime_cuda="$("${python_bin}" -c 'import torch; print(torch.version.cuda or "unknown")' 2>/dev/null)"; then
    echo "PyTorch runtime build CUDA: ${runtime_cuda} (${python_bin})"
  fi
}

activate_best_installed_compiler() {
  local candidate normalized release best_compiler="" best_release=""
  local seen_compilers=$'\n'
  local -a candidates=()

  [[ -n "${DG_JIT_NVCC_COMPILER:-}" ]] && candidates+=("${DG_JIT_NVCC_COMPILER}")
  candidates+=("${CUDA_TOOLKIT_ROOT}/bin/nvcc" "/usr/local/cuda/bin/nvcc")
  while IFS= read -r candidate; do
    [[ -n "${candidate}" ]] && candidates+=("${candidate}")
  done < <(compgen -G '/usr/local/cuda-*/bin/nvcc' 2>/dev/null || true)
  if command -v nvcc >/dev/null 2>&1; then
    candidates+=("$(command -v nvcc)")
  fi

  for candidate in "${candidates[@]}"; do
    [[ -d "${candidate}" ]] && candidate="${candidate}/bin/nvcc"
    [[ -x "${candidate}" ]] || continue
    normalized="$(readlink -f "${candidate}" 2>/dev/null || printf '%s' "${candidate}")"
    [[ "${seen_compilers}" == *$'\n'"${normalized}"$'\n'* ]] && continue
    seen_compilers+="${normalized}"$'\n'

    release="$(nvcc_release "${normalized}" || true)"
    nvcc_is_compatible "${release}" || continue
    echo "Detected compatible NVCC ${release}: ${normalized}"
    if [[ -z "${best_release}" ]] || nvcc_is_newer "${release}" "${best_release}"; then
      best_compiler="${normalized}"
      best_release="${release}"
    fi
  done

  [[ -n "${best_compiler}" ]] || return 1
  activate_compiler "${best_compiler}"
}

download_file() {
  local url="$1"
  local output="$2"
  local candidate python_bin=""
  if command -v curl >/dev/null 2>&1; then
    if curl -fsSL "${url}" -o "${output}"; then
      return 0
    fi
    rm -f "${output}"
    echo "curl download failed; trying another downloader." >&2
  fi
  if command -v wget >/dev/null 2>&1; then
    if wget -q "${url}" -O "${output}"; then
      return 0
    fi
    rm -f "${output}"
    echo "wget download failed; trying Python." >&2
  fi

  for candidate in \
    "${V4_PYTHON:-}" \
    "/opt/sglang-v4/bin/python" \
    "python3" \
    "python"; do
    [[ -n "${candidate}" ]] || continue
    if [[ "${candidate}" == */* ]]; then
      [[ -x "${candidate}" ]] || continue
      python_bin="${candidate}"
    elif command -v "${candidate}" >/dev/null 2>&1; then
      python_bin="$(command -v "${candidate}")"
    else
      continue
    fi
    break
  done
  [[ -n "${python_bin}" ]] || \
    fail "curl, wget, or a Python interpreter is required to add the NVIDIA CUDA repository"

  echo "Downloading with ${python_bin} standard library..."
  "${python_bin}" - "${url}" "${output}" <<'PY'
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen

url, output_name = sys.argv[1:]
output = Path(output_name)
partial = output.with_name(output.name + ".part")
request = Request(url, headers={"User-Agent": "EASYEP-CUDA-toolchain-installer/1"})

try:
    with urlopen(request, timeout=120) as response, partial.open("wb") as destination:
        while chunk := response.read(1024 * 1024):
            destination.write(chunk)
    os.replace(partial, output)
except Exception:
    partial.unlink(missing_ok=True)
    raise
PY
}

rewrite_cuda_repo_base() {
  local from_base="${1%/}"
  local to_base="${2%/}"
  local from_path source_file
  from_path="${from_base#*://}"

  while IFS= read -r source_file; do
    sed -i \
      -e "s#https://${from_path}#${to_base}#g" \
      -e "s#http://${from_path}#${to_base}#g" \
      "${source_file}"
  done < <(grep -RIlF "${from_path}" /etc/apt/sources.list.d 2>/dev/null || true)
}

apt_repository_ready() {
  local repo_base="$1"
  shift
  local apt_status=0 package update_log
  update_log="$(mktemp)"

  echo "Refreshing apt indexes for CUDA repository: ${repo_base}"
  apt-get update 2>&1 | tee "${update_log}" || apt_status="${PIPESTATUS[0]}"

  if (( apt_status != 0 )); then
    echo "CUDA repository health check failed: apt-get update exited ${apt_status}." >&2
    rm -f "${update_log}"
    return 1
  fi
  if grep -Eqi '(^Err:|Failed to fetch|Some index files failed|Could not handshake)' \
    "${update_log}"; then
    echo "CUDA repository health check failed: apt reported an index download error." >&2
    rm -f "${update_log}"
    return 1
  fi
  rm -f "${update_log}"

  for package in "$@"; do
    # Consume all apt-cache output; with pipefail, grep -q can make apt-cache
    # exit on SIGPIPE after the first matching package version.
    if ! apt-cache show "${package}" 2>/dev/null | grep '^Package:' >/dev/null; then
      echo "CUDA repository health check failed: package ${package} is not visible to apt." >&2
      return 1
    fi
  done
  return 0
}

install_toolchain() {
  local candidate repo_distro repo_arch keyring_url temp_dir
  local aliyun_image=0 selected_repo_base=""
  local alternate_repo_base=""
  local nvidia_global_base="https://developer.download.nvidia.com/compute/cuda/repos"
  local nvidia_china_base="https://developer.download.nvidia.cn/compute/cuda/repos"
  local -a packages repo_base_candidates

  report_cuda_context
  if activate_best_installed_compiler; then
    echo "Using the newest compatible CUDA compiler already installed; no package changes are needed."
    return 0
  fi

  [[ "$(id -u)" == "0" ]] || \
    fail "run this script as root inside the current Docker container"
  command -v apt-get >/dev/null 2>&1 || \
    fail "automatic repair currently supports Debian/Ubuntu apt-based Docker images"
  command -v dpkg >/dev/null 2>&1 || fail "dpkg was not found"
  command -v apt-cache >/dev/null 2>&1 || fail "apt-cache was not found"
  [[ -r /etc/os-release ]] || fail "/etc/os-release was not found"

  # shellcheck disable=SC1091
  . /etc/os-release
  case "${ID:-}:${VERSION_ID:-}" in
    ubuntu:20.04) repo_distro="ubuntu2004" ;;
    ubuntu:22.04) repo_distro="ubuntu2204" ;;
    ubuntu:24.04) repo_distro="ubuntu2404" ;;
    debian:11) repo_distro="debian11" ;;
    debian:12) repo_distro="debian12" ;;
    debian:13) repo_distro="debian13" ;;
    *) fail "unsupported apt base image: ID=${ID:-unknown} VERSION_ID=${VERSION_ID:-unknown}" ;;
  esac

  case "$(dpkg --print-architecture)" in
    amd64) repo_arch="x86_64" ;;
    arm64) repo_arch="sbsa" ;;
    *) fail "unsupported architecture: $(dpkg --print-architecture)" ;;
  esac

  if grep -RqiE 'mirrors\.(aliyun\.com|cloud\.aliyuncs\.com)|aliyun' \
    /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null; then
    aliyun_image=1
    echo "Detected Alibaba Cloud apt sources; preferring NVIDIA's China CDN."
  fi

  if [[ -n "${CUDA_REPO_BASE_URL}" ]]; then
    repo_base_candidates=("${CUDA_REPO_BASE_URL%/}")
    echo "Using explicit CUDA_REPO_BASE_URL override."
  elif [[ "${aliyun_image}" == "1" ]]; then
    repo_base_candidates=("${nvidia_china_base}" "${nvidia_global_base}")
  else
    repo_base_candidates=("${nvidia_global_base}" "${nvidia_china_base}")
  fi

  temp_dir="$(mktemp -d)"
  echo "Installing the NVIDIA CUDA repository keyring for ${repo_distro}/${repo_arch}..."
  for candidate in "${repo_base_candidates[@]}"; do
    keyring_url="${candidate}/${repo_distro}/${repo_arch}/cuda-keyring_1.1-1_all.deb"
    echo "Trying CUDA repository: ${candidate}"
    if download_file "${keyring_url}" "${temp_dir}/cuda-keyring.deb"; then
      selected_repo_base="${candidate}"
      break
    fi
    rm -f "${temp_dir}/cuda-keyring.deb"
    echo "Repository keyring download failed; trying the next endpoint." >&2
  done
  [[ -n "${selected_repo_base}" ]] || \
    fail "could not download the NVIDIA CUDA repository keyring from any configured endpoint"

  echo "Selected CUDA repository: ${selected_repo_base}"
  dpkg -i "${temp_dir}/cuda-keyring.deb"
  rm -rf "${temp_dir}"

  if [[ "${selected_repo_base}" != "${nvidia_global_base}" ]]; then
    rewrite_cuda_repo_base "${nvidia_global_base}" "${selected_repo_base}"
  fi

  read -r -a packages <<<"${CUDA_APT_PACKAGES}"
  if ! apt_repository_ready "${selected_repo_base}" "${packages[@]}"; then
    if [[ -n "${CUDA_REPO_BASE_URL}" ]]; then
      fail "explicit CUDA repository is unavailable: ${selected_repo_base}"
    fi

    if [[ "${selected_repo_base}" == "${nvidia_global_base}" ]]; then
      alternate_repo_base="${nvidia_china_base}"
    else
      alternate_repo_base="${nvidia_global_base}"
    fi
    echo "Switching CUDA repository to: ${alternate_repo_base}" >&2
    rewrite_cuda_repo_base "${selected_repo_base}" "${alternate_repo_base}"
    selected_repo_base="${alternate_repo_base}"
    apt_repository_ready "${selected_repo_base}" "${packages[@]}" || \
      fail "CUDA packages are unavailable from both NVIDIA repository endpoints"
  fi
  echo "CUDA apt repository ready: ${selected_repo_base}"
  echo "Installing CUDA JIT toolchain packages: ${packages[*]}"
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${packages[@]}"

  activate_compiler "${CUDA_TOOLKIT_ROOT}/bin/nvcc" || \
    fail "CUDA packages installed, but ${CUDA_TOOLKIT_ROOT}/bin/nvcc is missing or older than ${MIN_NVCC_MAJOR}.${MIN_NVCC_MINOR}"
}

if [[ "${EASYEP_TOOLCHAIN_LIB_ONLY:-0}" != "1" ]]; then
  install_toolchain
fi
