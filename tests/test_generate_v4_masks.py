from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in the local lightweight environment")
class GenerateV4MasksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pruning_dir = Path(__file__).resolve().parents[1] / "pruning"
        sys.path.insert(0, str(pruning_dir))
        path = pruning_dir / "generate_v4_masks.py"
        spec = importlib.util.spec_from_file_location("easyep_generate_v4_masks", path)
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_hash_rows_are_restored_for_every_target(self):
        scores = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
            ]
        )
        mask = self.module.v4_mask(scores, target_experts=2, hash_layers=1)
        self.assertEqual(mask[0], [1, 1, 1, 1])
        self.assertEqual(mask[1], [1, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
