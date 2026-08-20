#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class V4ArbitraryPruningScriptTests(unittest.TestCase):
    def test_generic_mask_entrypoint_aggregates_multiple_targets_once(self) -> None:
        source = (ROOT / "scripts/prepare_easyep_masks.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("TARGET_EXPERTS", source)
        self.assertIn("PRUNE_PERCENTAGES", source)
        self.assertEqual(source.count("pruning/generate_v4_masks.py"), 1)
        for forbidden in ("curl ", "wget ", "pip install", "uv pip", "apt-get"):
            self.assertNotIn(forbidden, source)

    def test_generic_materializer_reuses_existing_calibration(self) -> None:
        source = (
            ROOT / "scripts/prepare_v4_pruned_checkpoints_any.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('if [[ ! -f "${TOKEN_STATS}" ]]', source)
        self.assertIn("prepare_easyep_masks.sh", source)
        self.assertIn("--target-experts \"${target}\"", source)
        self.assertIn("--dry-run", source)
        self.assertIn("--verify-only", source)
        for forbidden in ("curl ", "wget ", "pip install", "uv pip", "apt-get"):
            self.assertNotIn(forbidden, source)

    def test_legacy_two_ratio_entrypoint_remains_available(self) -> None:
        source = (ROOT / "scripts/prepare_easyep_masks_25_50.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('TARGET_EXPERTS="192,128"', source)
        self.assertIn("prepare_easyep_masks.sh", source)


if __name__ == "__main__":
    unittest.main()
