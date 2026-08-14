#!/usr/bin/env bash
set -euo pipefail

# Read-only preflight for the in-place DeepSeek-V4-Flash Docker runtime.
#
# This file deliberately contains no apt, curl, wget, pip, uv, or Hugging Face
# download command.  Missing or incompatible prerequisites are reported and
# cause a failure; they are never installed automatically.

V4_VALIDATOR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V4_VALIDATOR_REPO_ROOT="$(cd "${V4_VALIDATOR_SCRIPT_DIR}/.." && pwd)"

v4_validation_fail() {
  echo "ERROR: $*" >&2
  echo "No download or installation was attempted." >&2
  return 1
}

v4_active_transfer_processes() {
  # Print only process metadata and executable names.  Command arguments are
  # intentionally excluded because they can contain credentials or URLs.
  ps -eo pid=,ppid=,etime=,comm= 2>/dev/null | awk '
    {
      name=$4
      sub(/^.*\//, "", name)
      if (name ~ /^(apt|apt-get|dpkg|curl|wget|pip|pip3|uv|aria2c|git-lfs|hf|huggingface-cli)$/)
        print
    }
  '
}

v4_nvcc_release() {
  local compiler="$1"
  "${compiler}" --version 2>/dev/null | sed -n \
    's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1.\2/p' | head -n 1
}

v4_activate_cuda13() {
  local candidate normalized release major minor
  local best_compiler="" best_release="" best_minor="-1"
  local seen_compilers=$'\n'
  local -a candidates=()

  [[ -n "${DG_JIT_NVCC_COMPILER:-}" ]] && \
    candidates+=("${DG_JIT_NVCC_COMPILER}")
  [[ -n "${CUDA_HOME:-}" ]] && candidates+=("${CUDA_HOME}/bin/nvcc")
  candidates+=("/usr/local/cuda-13.0/bin/nvcc" "/usr/local/cuda/bin/nvcc")
  while IFS= read -r candidate; do
    [[ -n "${candidate}" ]] && candidates+=("${candidate}")
  done < <(compgen -G '/usr/local/cuda-13*/bin/nvcc' 2>/dev/null || true)
  if command -v nvcc >/dev/null 2>&1; then
    candidates+=("$(command -v nvcc)")
  fi

  for candidate in "${candidates[@]}"; do
    [[ -d "${candidate}" ]] && candidate="${candidate}/bin/nvcc"
    [[ -x "${candidate}" ]] || continue
    normalized="$(readlink -f "${candidate}" 2>/dev/null || printf '%s' "${candidate}")"
    [[ "${seen_compilers}" == *$'\n'"${normalized}"$'\n'* ]] && continue
    seen_compilers+="${normalized}"$'\n'

    release="$(v4_nvcc_release "${normalized}" || true)"
    [[ "${release}" =~ ^([0-9]+)\.([0-9]+)$ ]] || continue
    major="${BASH_REMATCH[1]}"
    minor="${BASH_REMATCH[2]}"
    [[ "${major}" == "${EXPECTED_CUDA_MAJOR:-13}" ]] || continue
    if (( minor > best_minor )); then
      best_compiler="${normalized}"
      best_release="${release}"
      best_minor="${minor}"
    fi
  done

  [[ -n "${best_compiler}" ]] || \
    v4_validation_fail "CUDA ${EXPECTED_CUDA_MAJOR:-13}.x NVCC was not found inside the current container." || return 1

  export DG_JIT_NVCC_COMPILER="${best_compiler}"
  export CUDA_HOME="$(cd "$(dirname "${best_compiler}")/.." && pwd)"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  V4_VALIDATED_NVCC_VERSION="${best_release}"
  export V4_VALIDATED_NVCC_VERSION
  echo "CUDA compiler: NVCC ${best_release} (${best_compiler})"
}

v4_validate_local_model() {
  "${V4_PYTHON}" - "${MODEL_PATH}" <<'PY'
import json
from pathlib import Path
import sys

model = Path(sys.argv[1])
config = model / "config.json"
if not config.is_file():
    raise SystemExit(f"missing local model config: {config}")

indexes = sorted(model.glob("*.safetensors.index.json"))
if indexes:
    try:
        payload = json.loads(indexes[0].read_text(encoding="utf-8"))
        shards = sorted(set(payload["weight_map"].values()))
    except Exception as exc:
        raise SystemExit(f"invalid weight index {indexes[0]}: {exc}")
    missing = [name for name in shards if not (model / name).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise SystemExit(
            f"local checkpoint is incomplete: {len(missing)} shard(s) missing; first: {preview}"
        )
    print(f"Local model: config plus {len(shards)} indexed shard(s) present")
else:
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise SystemExit("no local safetensors index or weight shard was found")
    print(f"Local model: config plus {len(shards)} unindexed shard(s) present")
PY
}

v4_validate_python_runtime() {
  CUDA_VISIBLE_DEVICES="${V4_GPU_LIST:-4,5,6,7}" \
    "${V4_PYTHON}" - \
      "${EXPECTED_SGLANG_VERSION:-0.5.16}" \
      "${EXPECTED_TORCH_CUDA_MAJOR:-13}" \
      "${V4_VALIDATOR_REPO_ROOT}" <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

expected_sglang, expected_cuda_major, repo_root = sys.argv[1:]

try:
    sglang_version = version("sglang")
except PackageNotFoundError as exc:
    raise SystemExit("SGLang is not installed in the selected V4 Python") from exc
if sglang_version != expected_sglang:
    raise SystemExit(
        f"SGLang version mismatch: expected {expected_sglang}, found {sglang_version}"
    )

import sglang
import torch

module_file = sglang.__file__
if not module_file:
    raise SystemExit("SGLang resolved only as a namespace package")
module_path = Path(module_file).resolve()
legacy_path = (Path(repo_root) / "sglang").resolve()
if module_path == legacy_path or legacy_path in module_path.parents:
    raise SystemExit(f"repository legacy sglang shadows installed package: {module_path}")

torch_cuda = torch.version.cuda or ""
if torch_cuda.split(".", 1)[0] != expected_cuda_major:
    raise SystemExit(
        f"PyTorch CUDA build mismatch: expected {expected_cuda_major}.x, found {torch_cuda or 'none'}"
    )
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
if torch.cuda.device_count() != 4:
    raise SystemExit(
        f"CUDA_VISIBLE_DEVICES should expose 4 GPUs, found {torch.cuda.device_count()}"
    )

print(f"Python runtime: {sys.executable}")
print(f"SGLang: {sglang_version} ({module_path})")
print(f"PyTorch: {torch.__version__}; CUDA build: {torch_cuda}; visible GPUs: 4")
PY
}

v4_validate_sglang_cli() {
  local help_text required_flag
  if ! help_text="$(CUDA_VISIBLE_DEVICES="${V4_GPU_LIST:-4,5,6,7}" \
    "${V4_PYTHON}" -m sglang.launch_server --help 2>&1)"; then
    printf '%s\n' "${help_text}" >&2
    v4_validation_fail "SGLang launch_server help failed." || return 1
  fi
  for required_flag in \
    --moe-runner-backend \
    --reasoning-parser \
    --tool-call-parser \
    --watchdog-timeout \
    --disable-custom-all-reduce \
    --disable-shared-experts-fusion; do
    if ! grep -q -- "${required_flag}" <<<"${help_text}"; then
      v4_validation_fail "SGLang ${EXPECTED_SGLANG_VERSION:-0.5.16} lacks required option ${required_flag}." || return 1
    fi
  done
  echo "SGLang CLI: required DeepSeek-V4 launch options are available"
}

validate_v4_flash_runtime() {
  local active_transfers gpu_inventory gpu_index

  MODEL_PATH="${MODEL_PATH:-/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash}"
  V4_PYTHON="${V4_PYTHON:-/opt/sglang-v4/bin/python}"
  export MODEL_PATH V4_PYTHON

  # Prevent Transformers/Hugging Face code from filling in missing local files
  # from the network.  A missing checkpoint must fail the preflight instead.
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export HF_HUB_DISABLE_TELEMETRY=1

  echo "DeepSeek-V4-Flash runtime validation (read-only)"
  echo "Downloads/installers: disabled"
  echo "Hugging Face/Transformers offline mode: enabled"

  active_transfers="$(v4_active_transfer_processes || true)"
  if [[ -n "${active_transfers}" ]]; then
    echo "Active transfer/install process(es) detected (PID PPID ELAPSED COMMAND):" >&2
    printf '%s\n' "${active_transfers}" >&2
    if [[ "${IGNORE_ACTIVE_DOWNLOADS:-0}" != "1" ]]; then
      v4_validation_fail "Refusing to overlap with an active transfer/install process. Set IGNORE_ACTIVE_DOWNLOADS=1 only after verifying it is unrelated." || return 1
    fi
    echo "WARNING: active process check was explicitly overridden." >&2
  else
    echo "Active transfer/install process check: none detected"
  fi

  [[ -x "${V4_PYTHON}" ]] || \
    v4_validation_fail "V4 Python is missing or not executable: ${V4_PYTHON}" || return 1
  command -v nvidia-smi >/dev/null 2>&1 || \
    v4_validation_fail "nvidia-smi was not found inside the current container." || return 1

  gpu_inventory="$(nvidia-smi --query-gpu=index,name,memory.free \
    --format=csv,noheader,nounits)" || \
    v4_validation_fail "nvidia-smi failed." || return 1
  for gpu_index in 4 5 6 7; do
    if ! awk -F, -v expected="${gpu_index}" \
      '{ gpu_id=$1; gsub(/[[:space:]]/, "", gpu_id); if (gpu_id == expected) found=1 } END { exit !found }' \
      <<<"${gpu_inventory}"; then
      v4_validation_fail "physical GPU ${gpu_index} was not reported by nvidia-smi." || return 1
    fi
  done
  echo "Physical GPUs: 4,5,6,7 are present"

  v4_activate_cuda13 || return 1
  if ! v4_validate_python_runtime; then
    v4_validation_fail "Python/SGLang/PyTorch runtime validation failed." || return 1
  fi
  if ! v4_validate_local_model; then
    v4_validation_fail "The local model checkpoint is missing or incomplete." || return 1
  fi
  v4_validate_sglang_cli || return 1

  V4_VALIDATED_SGLANG_VERSION="${EXPECTED_SGLANG_VERSION:-0.5.16}"
  V4_VALIDATED_SGLANG_MODULE="$("${V4_PYTHON}" -c 'import sglang; print(sglang.__file__ or "")')"
  export V4_VALIDATED_SGLANG_VERSION V4_VALIDATED_SGLANG_MODULE

  echo "Validation passed. No apt, curl, wget, pip, uv, or model download command was executed."
}

if [[ "${V4_RUNTIME_VALIDATOR_LIB_ONLY:-0}" != "1" ]]; then
  validate_v4_flash_runtime
fi
