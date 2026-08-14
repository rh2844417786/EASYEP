from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in the local lightweight environment")
class ExpertSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "pruning" / "expert_selection.py"
        spec = importlib.util.spec_from_file_location("easyep_expert_selection", path)
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_dynamic_shape_and_per_layer_token_counts(self):
        sample = {
            "idxs": [[[0], [1]], [[2]]],
            "weights": [[[0.5], [0.25]], [[1.0]]],
            "norms": [[[2.0], [4.0]], [[3.0]]],
            "simibr": [[[0.0, 0.5]], [[0.25]]],
        }
        scores = self.module.aggregate_scores([sample], num_experts=3)
        self.assertEqual(tuple(scores.shape), (2, 3))
        self.assertAlmostEqual(scores[0, 0].item(), 1.0)
        self.assertAlmostEqual(scores[0, 1].item(), 0.5)
        self.assertAlmostEqual(scores[1, 2].item(), 2.25)


if __name__ == "__main__":
    unittest.main()
