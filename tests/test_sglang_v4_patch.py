#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "patch_sglang_v4_heterogeneous_experts.py"
)
SPEC = importlib.util.spec_from_file_location("easyep_sglang_v4_patch", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SglangV4PatchTests(unittest.TestCase):
    def _package(self, root: Path) -> Path:
        package = root / "sglang"
        config = package / MODULE.CONFIG_RELATIVE
        model = package / MODULE.MOE_RELATIVE
        config.parent.mkdir(parents=True)
        model.parent.mkdir(parents=True)
        config.write_text(
            "from typing import Dict, List, Optional\n"
            "class Config:\n"
            + MODULE.CONFIG_ANCHOR,
            encoding="utf-8",
        )
        model.write_text(
            "from __future__ import annotations\n"
            "class DeepseekV2MoE:\n"
            "    def __init__(self, config, layer_id, is_nextn, is_deepseek_v4):\n"
            + MODULE.MOE_INIT_ANCHOR
            + "        self.gate = MoEGate(config=config)\n"
            + "        pass\n"
            + "    def dual(self, hidden_states, gemm_output_zero_allocator):\n"
            + MODULE.GATE_PATCHES[0][0]
            + "        return router_logits\n"
            + "    def normal(self, hidden_states, gemm_output_zero_allocator):\n"
            + "        if True:\n"
            + MODULE.GATE_PATCHES[1][0]
            + "            return router_logits\n"
            + "    def cpu(self, hidden_states):\n"
            + MODULE.GATE_PATCHES[2][0]
            + "        return router_logits\n"
            + "    def deepep(self, hidden_states, forward_batch):\n"
            + "        if True:\n"
            + MODULE.GATE_PATCHES[3][0]
            + "            return router_logits\n"
            + "    def op_gate(self, state):\n"
            + "        if True:\n"
            + MODULE.GATE_PATCHES[4][0]
            + "            return state.router_logits\n"
            + MODULE.HELPER_ANCHOR
            + "        return False\n",
            encoding="utf-8",
        )
        return package

    def test_apply_check_idempotence_and_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = self._package(Path(tmp))

            MODULE.apply(package)
            MODULE.check(package)
            MODULE.apply(package)
            for path in MODULE.patch_specs(package):
                self.assertIn(MODULE.MARKER, path.read_text(encoding="utf-8"))
            config_source = (package / MODULE.CONFIG_RELATIVE).read_text(
                encoding="utf-8"
            )
            self.assertIn("easyep_expert_mask_by_layer", config_source)
            self.assertIn("easyep_pruning", config_source)
            moe_source = (package / MODULE.MOE_RELATIVE).read_text(encoding="utf-8")
            self.assertIn("_apply_easyep_expert_mask", moe_source)
            self.assertIn("self.gate = MoEGate", moe_source)
            self.assertIn("minimum_experts", moe_source)
            self.assertNotIn("must retain 128 or 192 experts", moe_source)

            MODULE.restore(package)
            for path in MODULE.patch_specs(package):
                self.assertNotIn(MODULE.MARKER, path.read_text(encoding="utf-8"))

    def test_unknown_source_is_rejected_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = self._package(Path(tmp))
            config = package / MODULE.CONFIG_RELATIVE
            original_config = config.read_text(encoding="utf-8")
            model = package / MODULE.MOE_RELATIVE
            original = "def incompatible():\n    return 1\n"
            model.write_text(original, encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "patch anchor"):
                MODULE.apply(package)
            self.assertEqual(model.read_text(encoding="utf-8"), original)
            self.assertEqual(config.read_text(encoding="utf-8"), original_config)

    def test_marker_without_complete_patch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = self._package(Path(tmp))
            MODULE.apply(package)
            config = package / MODULE.CONFIG_RELATIVE
            source = config.read_text(encoding="utf-8").replace(
                "    easyep_pruning: Optional[Dict[str, object]] = None\n", ""
            )
            config.write_text(source, encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "patch is incomplete"):
                MODULE.check(package)
            with self.assertRaisesRegex(RuntimeError, "patch is incomplete"):
                MODULE.apply(package)

    def test_apply_upgrades_existing_fixed_ratio_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = self._package(Path(tmp))
            MODULE.apply(package)
            model = package / MODULE.MOE_RELATIVE
            source = model.read_text(encoding="utf-8").replace(
                MODULE.MOE_INIT_REPLACEMENT,
                MODULE.LEGACY_MOE_INIT_REPLACEMENT,
                1,
            )
            model.write_text(source, encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "patch is incomplete"):
                MODULE.check(package)
            MODULE.apply(package)
            MODULE.check(package)
            upgraded = model.read_text(encoding="utf-8")
            self.assertIn(MODULE.MOE_INIT_REPLACEMENT, upgraded)
            self.assertNotIn(MODULE.LEGACY_MOE_INIT_REPLACEMENT, upgraded)


if __name__ == "__main__":
    unittest.main()
