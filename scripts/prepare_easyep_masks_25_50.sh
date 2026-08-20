#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible reproduction wrapper.  The generic entrypoint performs
# one score aggregation and emits both historical 25% and 50% masks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET_EXPERTS="192,128" \
  bash "${SCRIPT_DIR}/prepare_easyep_masks.sh"
