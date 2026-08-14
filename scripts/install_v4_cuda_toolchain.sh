#!/usr/bin/env bash
set -euo pipefail

# Install a DeepGEMM-compatible CUDA compiler into the current Docker image.
# This intentionally installs toolkit/compiler packages only; it never installs
# or replaces the host NVIDIA driver.

CUDA_TOOLKIT_SERIES="${CUDA_TOOLKIT_SERIES:-12-9}"
CUDA_TOOLKIT_DOT="${CUDA_TOOLKIT_DOT:-12.9}"
CUDA_TOOLKIT_ROOT="${CUDA_TOOLKIT_ROOT:-/usr/local/cuda-${CUDA_TOOLKIT_DOT}}"
CUDA_APT_PACKAGES="${CUDA_APT_PACKAGES:-cuda-compiler-${CUDA_TOOLKIT_SERIES} cuda-cudart-dev-${CUDA_TOOLKIT_SERIES} cuda-cccl-${CUDA_TOOLKIT_SERIES}}"
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
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${url}" -o "${output}"
  elif command -v wget >/dev/null 2>&1; then
    wget -q "${url}" -O "${output}"
  else
    fail "curl or wget is required to add the NVIDIA CUDA repository"
  fi
}

install_toolchain() {
  local candidate repo_distro repo_arch keyring_url temp_dir
  local -a packages

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

  temp_dir="$(mktemp -d)"
  keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/${repo_distro}/${repo_arch}/cuda-keyring_1.1-1_all.deb"
  echo "Installing the NVIDIA CUDA repository keyring for ${repo_distro}/${repo_arch}..."
  download_file "${keyring_url}" "${temp_dir}/cuda-keyring.deb"
  dpkg -i "${temp_dir}/cuda-keyring.deb"
  rm -rf "${temp_dir}"

  apt-get update
  read -r -a packages <<<"${CUDA_APT_PACKAGES}"
  echo "Installing CUDA JIT toolchain packages: ${packages[*]}"
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${packages[@]}"

  activate_compiler "${CUDA_TOOLKIT_ROOT}/bin/nvcc" || \
    fail "CUDA packages installed, but ${CUDA_TOOLKIT_ROOT}/bin/nvcc is missing or older than ${MIN_NVCC_MAJOR}.${MIN_NVCC_MINOR}"
}

install_toolchain
