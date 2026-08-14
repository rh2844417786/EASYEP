#!/usr/bin/env bash
set -euo pipefail

# Install a DeepGEMM-compatible CUDA compiler into the current Docker image.
# This intentionally installs toolkit/compiler packages only; it never installs
# or replaces the host NVIDIA driver.

CUDA_TOOLKIT_SERIES="${CUDA_TOOLKIT_SERIES:-12-9}"
CUDA_TOOLKIT_DOT="${CUDA_TOOLKIT_DOT:-12.9}"
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

install_toolchain() {
  local candidate repo_distro repo_arch keyring_url temp_dir source_file
  local aliyun_image=0 selected_repo_base=""
  local nvidia_global_base="https://developer.download.nvidia.com/compute/cuda/repos"
  local nvidia_china_base="https://developer.download.nvidia.cn/compute/cuda/repos"
  local -a packages repo_base_candidates

  for candidate in \
    "${DG_JIT_NVCC_COMPILER:-}" \
    "${CUDA_TOOLKIT_ROOT}/bin/nvcc" \
    "/usr/local/cuda/bin/nvcc" \
    "$(command -v nvcc 2>/dev/null || true)"; do
    [[ -n "${candidate}" ]] || continue
    [[ -d "${candidate}" ]] && candidate="${candidate}/bin/nvcc"
    if activate_compiler "${candidate}"; then
      echo "A compatible CUDA compiler is already installed; no package changes are needed."
      return 0
    fi
  done

  [[ "$(id -u)" == "0" ]] || \
    fail "run this script as root inside the current Docker container"
  command -v apt-get >/dev/null 2>&1 || \
    fail "automatic repair currently supports Debian/Ubuntu apt-based Docker images"
  command -v dpkg >/dev/null 2>&1 || fail "dpkg was not found"
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
    while IFS= read -r source_file; do
      sed -i \
        -e "s#https://developer.download.nvidia.com/compute/cuda/repos#${selected_repo_base}#g" \
        -e "s#http://developer.download.nvidia.com/compute/cuda/repos#${selected_repo_base}#g" \
        "${source_file}"
    done < <(grep -RIl 'developer\.download\.nvidia\.com/compute/cuda/repos' \
      /etc/apt/sources.list.d 2>/dev/null || true)
  fi

  if ! apt-get update; then
    if [[ -z "${CUDA_REPO_BASE_URL}" && "${selected_repo_base}" == "${nvidia_china_base}" ]]; then
      echo "NVIDIA China CDN apt update failed; retrying the global NVIDIA repository." >&2
      while IFS= read -r source_file; do
        sed -i "s#${nvidia_china_base}#${nvidia_global_base}#g" "${source_file}"
      done < <(grep -RIl "${nvidia_china_base}" /etc/apt/sources.list.d 2>/dev/null || true)
      selected_repo_base="${nvidia_global_base}"
      apt-get update
    else
      fail "apt-get update failed for CUDA repository ${selected_repo_base}"
    fi
  fi
  echo "CUDA apt repository ready: ${selected_repo_base}"
  read -r -a packages <<<"${CUDA_APT_PACKAGES}"
  echo "Installing CUDA JIT toolchain packages: ${packages[*]}"
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${packages[@]}"

  activate_compiler "${CUDA_TOOLKIT_ROOT}/bin/nvcc" || \
    fail "CUDA packages installed, but ${CUDA_TOOLKIT_ROOT}/bin/nvcc is missing or older than ${MIN_NVCC_MAJOR}.${MIN_NVCC_MINOR}"
}

if [[ "${EASYEP_TOOLCHAIN_LIB_ONLY:-0}" != "1" ]]; then
  install_toolchain
fi
