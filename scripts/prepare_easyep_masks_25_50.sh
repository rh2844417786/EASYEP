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
NUM_LAYERS="${NUM_LAYERS:-43}"
HASH_LAYERS="${HASH_LAYERS:-3}"
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
[[ "${NUM_LAYERS}" == "43" && "${HASH_LAYERS}" == "3" ]] || \
  fail "this V4-Flash protocol requires 43 layers with the first 3 hash layers preserved"

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
    if len(mask) != 43 or any(len(row) != 256 for row in mask):
        raise SystemExit(f"{path} must have 43 rows of 256 experts")
    # The first three layers use frozen token-id -> expert-id hash tables.
    # Preserve every expert there; only layers 3..42 are pruning candidates.
    for layer in range(3):
        mask[layer] = [1] * 256
    path.write_text(
        json.dumps(mask, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    counts = [sum(1 for value in row if value == 1) for row in mask]
    if counts[:3] != [256, 256, 256]:
        raise SystemExit(f"{path} modified a hash-routed layer")
    if any(count != expected_keep for count in counts[3:]):
        raise SystemExit(
            f"{path} dynamic-layer keep counts are invalid: {sorted(set(counts[3:]))}"
        )
    records.append(
        {
            "path": str(path),
            "dynamic_layer_prune_percent": prune_percent,
            "hash_layers_preserved": [0, 1, 2],
            "dynamic_layers_pruned": list(range(3, 43)),
            "hash_layer_experts": 256,
            "dynamic_layer_experts": expected_keep,
            "main_layer_slot_prune_percent": (1 - sum(counts) / (43 * 256)) * 100,
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
        "Masks are not pruned checkpoints. Layers 0..2 remain at 256 experts; "
        "layers 3..42 require the validated EASY-EP mask-routing runtime."
    ),
}
output = Path(mask25_name).parent / "aime_v4_mask_manifest.json"
output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"mask manifest: {output}")
PY

echo "Generated:"
echo "  25% dynamic-layer mask (layers 0..2 keep 256; layers 3..42 keep 192): ${mask25}"
echo "  50% dynamic-layer mask (layers 0..2 keep 256; layers 3..42 keep 128): ${mask50}"
echo "These are masks only; no V4 checkpoint was physically pruned."
