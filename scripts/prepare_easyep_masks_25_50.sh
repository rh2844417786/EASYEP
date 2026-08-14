#!/usr/bin/env bash
set -euo pipefail

# Build the two EASY-EP masks used by the reproduction matrix.
# This produces routing masks only. It does not create pruned checkpoints.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
V4_PYTHON="${V4_PYTHON:-/opt/sglang-v4/bin/python}"
TOKEN_STATS="${TOKEN_STATS:-${REPO_ROOT}/expert_statistics/token_information/aime_v4.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/expert_statistics/expert_mask}"
NUM_EXPERTS="${NUM_EXPERTS:-256}"
NUM_SAMPLES="${NUM_SAMPLES:-25}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-longest}"
SEED="${SEED:-42}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -x "${V4_PYTHON}" ]] || fail "V4 Python is not executable: ${V4_PYTHON}"
[[ -f "${TOKEN_STATS}" ]] || \
  fail "probe statistics not found: ${TOKEN_STATS}; complete Gate 2/3 in docs/deepseek_v4_flash_reproduction.md first"
[[ "${NUM_EXPERTS}" == "256" ]] || \
  fail "this reproduction protocol expects the repository's 256-expert checkpoint"

mkdir -p "${OUTPUT_DIR}"
scores_file="${OUTPUT_DIR}/aime_v4_scores.pt"
mask25="${OUTPUT_DIR}/aime_v4_prune25_keep192.json"
mask50="${OUTPUT_DIR}/aime_v4_prune50_keep128.json"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
started_seconds="${SECONDS}"

"${V4_PYTHON}" "${REPO_ROOT}/pruning/expert_selection.py" \
  --input-file "${TOKEN_STATS}" \
  --output-file "${scores_file}" \
  --expert-mask "${mask25}" \
  --target-number 192 \
  --num-experts "${NUM_EXPERTS}" \
  --num-samples "${NUM_SAMPLES}" \
  --sample-strategy "${SAMPLE_STRATEGY}" \
  --seed "${SEED}"

"${V4_PYTHON}" "${REPO_ROOT}/pruning/expert_selection.py" \
  --input-file "${TOKEN_STATS}" \
  --output-file "${scores_file}" \
  --expert-mask "${mask50}" \
  --target-number 128 \
  --num-experts "${NUM_EXPERTS}" \
  --num-samples "${NUM_SAMPLES}" \
  --sample-strategy "${SAMPLE_STRATEGY}" \
  --seed "${SEED}"

"${V4_PYTHON}" - \
  "${TOKEN_STATS}" "${scores_file}" "${mask25}" "${mask50}" \
  "${started_at}" "$((SECONDS - started_seconds))" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

stats_name, scores_name, mask25_name, mask50_name, started_at, elapsed = sys.argv[1:]
stats = Path(stats_name)
scores = Path(scores_name)
mask_paths = [(Path(mask25_name), 192, 25), (Path(mask50_name), 128, 50)]
records = []
for path, expected_keep, prune_percent in mask_paths:
    mask = json.loads(path.read_text(encoding="utf-8"))
    if not mask or any(len(row) != 256 for row in mask):
        raise SystemExit(f"{path} must have non-empty rows of 256 experts")
    counts = [sum(1 for value in row if value == 1) for row in mask]
    if any(count != expected_keep for count in counts):
        raise SystemExit(f"{path} keep counts are invalid: {sorted(set(counts))}")
    records.append(
        {
            "path": str(path),
            "prune_percent": prune_percent,
            "keep_experts_per_layer": expected_keep,
            "layers": len(mask),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
manifest = {
    "type": "easyep_mask_manifest",
    "started_at": started_at,
    "elapsed_seconds": int(elapsed),
    "token_statistics": str(stats),
    "token_statistics_sha256": hashlib.sha256(stats.read_bytes()).hexdigest(),
    "scores": str(scores),
    "masks": records,
    "physical_pruned_checkpoints_created": False,
    "warning": (
        "Masks are not pruned checkpoints. DeepSeek-V4 hash routing needs a validated "
        "token-to-expert remap or heterogeneous expert runtime before evaluation."
    ),
}
output = Path(mask25_name).parent / "aime_v4_mask_manifest.json"
output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"mask manifest: {output}")
PY

echo "Generated:"
echo "  25% prune mask (keep 192/256): ${mask25}"
echo "  50% prune mask (keep 128/256): ${mask50}"
echo "These are masks only; no V4 checkpoint was physically pruned."
