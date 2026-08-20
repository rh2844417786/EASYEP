#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "pruning" / "model_prune_v4.py"
SPEC = importlib.util.spec_from_file_location("easyep_model_prune_v4", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def v4_config() -> dict:
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "num_hash_layers": 3,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
    }


def mask_for(target: int) -> list[list[int]]:
    mask = [[1] * 256 for _ in range(3)]
    selected = set(range(256 - target, 256))
    row = [int(expert in selected) for expert in range(256)]
    mask.extend([list(row) for _ in range(40)])
    return mask


def synthetic_index() -> dict:
    weight_map: dict[str, str] = {"embed.weight": "model-1.safetensors"}
    for layer in range(43):
        for expert in range(256):
            weight_map[
                f"layers.{layer}.ffn.experts.{expert}.w1.weight"
            ] = "model-1.safetensors"
        weight_map[f"layers.{layer}.ffn.gate.weight"] = "model-1.safetensors"
        if layer < 3:
            weight_map[f"layers.{layer}.ffn.gate.tid2eid"] = "model-1.safetensors"
        else:
            weight_map[f"layers.{layer}.ffn.gate.bias"] = "model-1.safetensors"
    return {"metadata": {"total_size": 1}, "weight_map": weight_map}


class V4PhysicalPruningTests(unittest.TestCase):
    def test_layout_preserves_hash_layers_and_prunes_only_dynamic_layers(self) -> None:
        layout = MODULE.build_layout(v4_config(), mask_for(192), 192)

        self.assertEqual(layout.counts_by_layer[:3], (256, 256, 256))
        self.assertEqual(set(layout.counts_by_layer[3:]), {192})
        self.assertAlmostEqual(layout.dynamic_prune_fraction, 0.25)
        self.assertAlmostEqual(
            layout.main_layer_slot_prune_fraction,
            1 - (3 * 256 + 40 * 192) / (43 * 256),
        )

    def test_layout_accepts_arbitrary_dynamic_expert_count(self) -> None:
        layout = MODULE.build_layout(v4_config(), mask_for(160), 160)

        self.assertEqual(layout.counts_by_layer[:3], (256, 256, 256))
        self.assertEqual(set(layout.counts_by_layer[3:]), {160})
        self.assertAlmostEqual(layout.dynamic_prune_fraction, 0.375)

    def test_layout_accepts_valid_boundaries(self) -> None:
        for target in (6, 256):
            with self.subTest(target=target):
                layout = MODULE.build_layout(v4_config(), mask_for(target), target)
                self.assertEqual(set(layout.counts_by_layer[3:]), {target})

    def test_layout_rejects_less_than_router_topk(self) -> None:
        with self.assertRaisesRegex(ValueError, r"per-token Top-K \(6\)"):
            MODULE.build_layout(v4_config(), mask_for(5), 5)
        with self.assertRaisesRegex(ValueError, "and 256"):
            MODULE.build_layout(v4_config(), mask_for(257), 257)

    def test_parser_accepts_non_historical_target(self) -> None:
        args = MODULE.build_parser().parse_args(
            [
                "--input-dir",
                "/tmp/input",
                "--output-dir",
                "/tmp/output",
                "--mask-json",
                "/tmp/mask.json",
                "--target-experts",
                "160",
            ]
        )
        self.assertEqual(args.target_experts, 160)

    def test_hash_layer_mask_change_is_rejected(self) -> None:
        mask = mask_for(128)
        mask[1][255] = 0

        with self.assertRaisesRegex(ValueError, "hash layer 1"):
            MODULE.build_layout(v4_config(), mask, 128)

    def test_plan_renumbers_dynamic_experts_but_preserves_router_and_hash(self) -> None:
        layout = MODULE.build_layout(v4_config(), mask_for(192), 192)
        plan = MODULE.build_plan(
            synthetic_index(),
            layout,
            provenance={
                "source_config_sha256": "config",
                "source_index_sha256": "index",
            },
        )

        hash_key = "layers.0.ffn.experts.255.w1.weight"
        dropped = "layers.3.ffn.experts.0.w1.weight"
        retained = "layers.3.ffn.experts.64.w1.weight"
        self.assertEqual(plan.key_actions[hash_key], hash_key)
        self.assertIsNone(plan.key_actions[dropped])
        self.assertEqual(
            plan.key_actions[retained],
            "layers.3.ffn.experts.0.w1.weight",
        )
        self.assertEqual(
            plan.key_actions["layers.3.ffn.gate.weight"],
            "layers.3.ffn.gate.weight",
        )
        self.assertEqual(
            plan.key_actions["layers.3.ffn.gate.bias"],
            "layers.3.ffn.gate.bias",
        )

    def test_output_config_declares_mask_routing_runtime_contract(self) -> None:
        config = v4_config()
        layout = MODULE.build_layout(config, mask_for(128), 128)
        plan = MODULE.build_plan(
            synthetic_index(),
            layout,
            provenance={
                "source_config_sha256": "config",
                "source_index_sha256": "index",
            },
        )

        output = MODULE.make_output_config(config, plan, "mask-sha")

        self.assertEqual(output["n_routed_experts"], 256)
        mask = output["easyep_expert_mask_by_layer"]
        self.assertEqual([sum(row) for row in mask[:3]], [256, 256, 256])
        self.assertEqual(set(sum(row) for row in mask[3:]), {128})
        self.assertTrue(output["easyep_pruning"]["hash_layers_preserved"])
        self.assertFalse(output["easyep_pruning"]["router_parameters_pruned"])
        self.assertTrue(output["easyep_pruning"]["router_mask_applied_at_runtime"])
        self.assertFalse(output["easyep_pruning"]["mtp_pruned"])


if __name__ == "__main__":
    unittest.main()
