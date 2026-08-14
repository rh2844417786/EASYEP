from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import v4_preflight


OFFICIAL_CONFIG = {
    "model_type": "deepseek_v4",
    "hidden_size": 4096,
    "num_hidden_layers": 43,
    "num_hash_layers": 3,
    "n_routed_experts": 256,
    "num_experts_per_tok": 6,
    "scoring_func": "sqrtsoftplus",
    "expert_dtype": "fp4",
}


class V4PreflightTests(unittest.TestCase):
    def make_args(self, root: Path, *, tp: int = 8, backend: str = "marlin"):
        return argparse.Namespace(
            model_path=root,
            config=None,
            tp=tp,
            backend=backend,
            min_free_mib=70_000,
            allow_no_gpu=False,
            json=False,
        )

    def test_h100_tp8_marlin_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(OFFICIAL_CONFIG))
            gpus = [
                {"index": index, "name": "NVIDIA H100 80GB HBM3", "memory.total": 81559, "memory.free": 80000}
                for index in range(8)
            ]
            with mock.patch.object(v4_preflight, "query_gpus", return_value=(gpus, None)), mock.patch.object(
                v4_preflight, "installed_version", return_value="0.5.15"
            ), mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}):
                result = v4_preflight.run_checks(self.make_args(root))
            self.assertTrue(result["ok"], result)

    def test_h100_tp4_triton_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(OFFICIAL_CONFIG))
            gpus = [
                {"index": index, "name": "NVIDIA H100", "memory.total": 81559, "memory.free": 80000}
                for index in range(4)
            ]
            with mock.patch.object(v4_preflight, "query_gpus", return_value=(gpus, None)), mock.patch.object(
                v4_preflight, "installed_version", return_value="0.5.15"
            ), mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}):
                result = v4_preflight.run_checks(self.make_args(root, tp=4, backend="triton"))
            self.assertFalse(result["ok"])
            self.assertTrue(any("TP=8" in error for error in result["errors"]))
            self.assertTrue(any("hidden-size mismatch" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
