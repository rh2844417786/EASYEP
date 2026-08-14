from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


try:
    import torch
except ImportError:  # local macOS workspace intentionally has no ML runtime
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in the local lightweight environment")
class ModelPruneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "pruning" / "model_prune.py"
        spec = importlib.util.spec_from_file_location("easyep_model_prune", path)
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_hash_routing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "hash-routed"):
            self.module.validate_v4_hash_routing({"num_hash_layers": 3}, {})

    def test_dynamic_layers_and_renumbering(self):
        weight_map = {
            f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight": "model-1.safetensors"
            for layer in (2, 5)
            for expert in range(4)
        }
        weight_map["model.embed_tokens.weight"] = "model-1.safetensors"
        layers, experts = self.module.inspect_layout(weight_map)
        mask = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.bool)
        filtered, renames, kept = self.module.build_renames(weight_map, mask, layers, experts)
        self.assertEqual(layers, [2, 5])
        self.assertEqual(kept, 2)
        old = "model.layers.2.mlp.experts.3.gate_proj.weight"
        self.assertEqual(renames[old], "model.layers.2.mlp.experts.1.gate_proj.weight")
        self.assertNotIn("model.layers.2.mlp.experts.0.gate_proj.weight", filtered)


if __name__ == "__main__":
    unittest.main()
