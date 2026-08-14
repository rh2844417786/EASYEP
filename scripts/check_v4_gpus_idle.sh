#!/usr/bin/env bash
set -euo pipefail

# Read-only exclusivity gate for the four physical GPUs used by the V4
# reproduction.  It never terminates a process.  A busy GPU is reported with
# the best process details available inside the current PID namespace.

GPU_LIST="${GPU_LIST:-4,5,6,7}"
MAX_PREEXISTING_GPU_MEMORY_MIB="${MAX_PREEXISTING_GPU_MEMORY_MIB:-2048}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ "${GPU_LIST}" == "4,5,6,7" ]] || \
  fail "this check is restricted to physical GPUs 4,5,6,7"
[[ "${MAX_PREEXISTING_GPU_MEMORY_MIB}" =~ ^[0-9]+$ ]] || \
  fail "MAX_PREEXISTING_GPU_MEMORY_MIB must be a non-negative integer"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi was not found"

inventory="$(nvidia-smi -i "${GPU_LIST}" \
  --query-gpu=index,memory.used \
  --format=csv,noheader,nounits)" || fail "nvidia-smi GPU memory query failed"

seen_gpus=()
seen_memory=()
while IFS=',' read -r raw_gpu raw_used; do
  gpu="${raw_gpu//[!0-9]/}"
  used="${raw_used//[!0-9]/}"
  [[ -n "${gpu}" && -n "${used}" ]] || \
    fail "unexpected nvidia-smi row: ${raw_gpu},${raw_used}"
  seen_gpus+=("${gpu}")
  seen_memory+=("${used}")
done <<<"${inventory}"

for expected_gpu in 4 5 6 7; do
  found=0
  for gpu in "${seen_gpus[@]}"; do
    [[ "${gpu}" == "${expected_gpu}" ]] && found=$((found + 1))
  done
  [[ "${found}" -eq 1 ]] || \
    fail "physical GPU ${expected_gpu} was reported ${found} time(s), expected exactly once"
done
[[ "${#seen_gpus[@]}" -eq 4 ]] || \
  fail "nvidia-smi returned ${#seen_gpus[@]} rows for GPUs ${GPU_LIST}, expected 4"

# A compute process is always considered busy, even when its context currently
# holds less memory than the baseline threshold. The threshold is only a
# fallback for memory that nvidia-smi cannot attribute inside this PID namespace.
busy_gpus=()
for index in "${!seen_gpus[@]}"; do
  gpu="${seen_gpus[${index}]}"
  used="${seen_memory[${index}]}"
  if ! process_rows="$(nvidia-smi -i "${gpu}" \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null)"; then
    fail "cannot establish exclusivity because the compute-process query failed for GPU ${gpu}"
  fi
  has_compute_process=0
  while IFS=',' read -r raw_pid _rest; do
    pid="${raw_pid//[!0-9]/}"
    [[ -n "${pid}" ]] && has_compute_process=1
  done <<<"${process_rows}"
  if (( has_compute_process == 1 )); then
    busy_gpus+=("${gpu}:${used}:compute-process")
  elif (( used > MAX_PREEXISTING_GPU_MEMORY_MIB )); then
    busy_gpus+=("${gpu}:${used}:unattributed-memory")
  fi
done

if [[ "${#busy_gpus[@]}" -eq 0 ]]; then
  echo "GPU exclusivity check: PASS (GPUs ${GPU_LIST}; no compute processes; each uses at most ${MAX_PREEXISTING_GPU_MEMORY_MIB} MiB)"
  printf '%s\n' "${inventory}"
  exit 0
fi

echo "ERROR: selected GPUs are already occupied before the V4 model launch." >&2
echo "Allowed pre-existing memory per GPU: ${MAX_PREEXISTING_GPU_MEMORY_MIB} MiB" >&2
echo "Observed GPU memory:" >&2
printf '%s\n' "${inventory}" >&2
echo "Busy GPU details (read-only; no process was terminated):" >&2

for item in "${busy_gpus[@]}"; do
  gpu="${item%%:*}"
  remainder="${item#*:}"
  used="${remainder%%:*}"
  reason="${item##*:}"
  echo "GPU ${gpu}: ${used} MiB used; reason=${reason}" >&2
  process_rows="$(nvidia-smi -i "${gpu}" \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "${process_rows}" ]]; then
    echo "  nvidia-smi exposed no compute-process details in this container." >&2
    continue
  fi
  while IFS=',' read -r raw_pid raw_name raw_memory; do
    pid="${raw_pid//[!0-9]/}"
    [[ -n "${pid}" ]] || continue
    name="${raw_name#${raw_name%%[![:space:]]*}}"
    memory="${raw_memory//[!0-9]/}"
    echo "  PID ${pid}: ${name:-unknown}; GPU memory=${memory:-unknown} MiB" >&2
    if command -v ps >/dev/null 2>&1; then
      ps -p "${pid}" -o pid=,ppid=,etimes=,user=,args= 2>/dev/null | \
        sed 's/^/    process: /' >&2 || true
    fi
    if [[ -r "/proc/${pid}/status" ]]; then
      parent_pid="$(awk '/^PPid:/ {print $2}' "/proc/${pid}/status" 2>/dev/null || true)"
      if [[ -n "${parent_pid}" && "${parent_pid}" != "0" ]] && command -v ps >/dev/null 2>&1; then
        ps -p "${parent_pid}" -o pid=,ppid=,etimes=,user=,args= 2>/dev/null | \
          sed 's/^/    parent:  /' >&2 || true
      fi
    fi
  done <<<"${process_rows}"
done

echo "Refusing to start another V4 process group on GPUs ${GPU_LIST}." >&2
echo "Stop or intentionally reassign the listed workload, then rerun the pipeline." >&2
exit 2
