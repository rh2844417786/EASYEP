#!/usr/bin/env python3

from __future__ import annotations

from decimal import Decimal
import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "pruning" / "v4_pruning_targets.py"
SPEC = importlib.util.spec_from_file_location("easyep_v4_pruning_targets", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class V4PruningTargetTests(unittest.TestCase):
    def test_historical_percentages_resolve_exactly(self) -> None:
        self.assertEqual(MODULE.target_from_prune_percent("25", 256), 192)
        self.assertEqual(MODULE.target_from_prune_percent("50", 256), 128)

    def test_fractional_target_rounds_retention_up(self) -> None:
        target = MODULE.target_from_prune_percent("30", 256)
        self.assertEqual(target, 180)
        self.assertEqual(
            MODULE.actual_prune_percent(target, 256), Decimal("29.6875")
        )

    def test_targets_are_deduplicated_in_request_order(self) -> None:
        self.assertEqual(
            MODULE.resolve_targets([160, 192], ["37.5", "50"], 256),
            [160, 192, 128],
        )

    def test_invalid_percentage_and_target_are_rejected(self) -> None:
        for value in ("-1", "100", "nan", "inf"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    MODULE.target_from_prune_percent(value, 256)
        with self.assertRaises(ValueError):
            MODULE.resolve_targets([0], [], 256)

    def test_stable_artifact_names_use_actual_discrete_rate(self) -> None:
        self.assertEqual(
            MODULE.mask_filename("aime_v4", 192, 256),
            "aime_v4_prune25_keep192.json",
        )
        self.assertEqual(
            MODULE.model_directory_name(180, 256),
            "v4-prune29p6875-keep180",
        )

    def test_prefix_rejects_paths(self) -> None:
        self.assertEqual(MODULE.validate_safe_prefix("mixed_v4"), "mixed_v4")
        for value in ("", "../bad", "bad/name", "bad name"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    MODULE.validate_safe_prefix(value)


if __name__ == "__main__":
    unittest.main()
