#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "evaluation" / "run_reproduction_matrix.py"
)
SPEC = importlib.util.spec_from_file_location("easyep_reproduction_matrix", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ReproductionMatrixTests(unittest.TestCase):
    def _checkpoint(
        self,
        root: Path,
        name: str,
        experts: int,
        *,
        hash_layers: int = 0,
        marker: str | None = None,
    ) -> Path:
        path = root / name
        path.mkdir()
        config = {
            "n_routed_experts": experts,
            "num_hash_layers": hash_layers,
            "test_marker": marker if marker is not None else name,
        }
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (path / "model-00001-of-00001.safetensors").write_bytes(name.encode())
        index = {
            "metadata": {},
            "weight_map": {
                "model.layers.0.mlp.experts.0.weight": (
                    "model-00001-of-00001.safetensors"
                )
            },
        }
        (path / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        return path

    def test_variant_matrix_requires_256_192_128_distinct_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = MODULE.build_variant_specs(
                self._checkpoint(root, "full", 256),
                self._checkpoint(root, "p25", 192),
                self._checkpoint(root, "p50", 128),
            )

            self.assertEqual([spec.name for spec in specs], ["full", "prune25", "prune50"])
            self.assertEqual([spec.expected_experts for spec in specs], [256, 192, 128])
            self.assertEqual([spec.prune_fraction for spec in specs], [0.0, 0.25, 0.5])

    def test_wrong_pruning_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "prune25 must have"):
                MODULE.build_variant_specs(
                    self._checkpoint(root, "full", 256),
                    self._checkpoint(root, "p25", 256),
                    self._checkpoint(root, "p50", 128),
                )

    def test_hash_routed_v4_pruning_is_rejected_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = self._checkpoint(root, "full", 256, hash_layers=3)
            p25 = self._checkpoint(root, "p25", 192, hash_layers=3)
            p50 = self._checkpoint(root, "p50", 128, hash_layers=3)
            with self.assertRaisesRegex(ValueError, "hash-routed"):
                MODULE.build_variant_specs(full, p25, p50)

            specs = MODULE.build_variant_specs(
                full,
                p25,
                p50,
                allow_hash_routed_pruned_checkpoints=True,
            )
            self.assertEqual(len(specs), 3)

    def test_expected_expert_counts_produce_distinct_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = self._checkpoint(root, "full", 256, marker="same")
            p25 = self._checkpoint(root, "p25", 192, marker="same")
            p50 = self._checkpoint(root, "p50", 128, marker="same")
            # Different expert counts make the config fingerprints distinct.
            specs = MODULE.build_variant_specs(full, p25, p50)
            self.assertEqual(len({spec.checkpoint.fingerprint for spec in specs}), 3)

    def test_parse_evaluation_output_preserves_accuracy_time_and_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "AIME24.jsonl"
            records = [
                {
                    "type": "sample",
                    "latency_seconds": 1.0,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
                {
                    "type": "sample",
                    "latency_seconds": 3.0,
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                },
                {
                    "type": "summary",
                    "accuracy": 50.0,
                    "correct": 1,
                    "total": 2,
                    "evaluator_backend": "test",
                },
            ]
            output.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            parsed = MODULE.parse_evaluation_output(output, 8.0)

            self.assertEqual(parsed["accuracy"], 50.0)
            self.assertEqual(parsed["latency_mean_seconds"], 2.0)
            self.assertEqual(parsed["latency_p95_seconds"], 3.0)
            self.assertEqual(parsed["completion_tokens"], 12)
            self.assertEqual(parsed["wall_seconds"], 8.0)

    def test_gpu_summary_reports_per_gpu_peaks(self) -> None:
        monitor = MODULE.GpuMonitor(Path("unused.csv"), "full", "4,5", 1.0)
        monitor.samples = [
            {
                "gpu_index": 4,
                "gpu_name": "H100",
                "memory_used_mib": 100.0,
                "utilization_gpu_percent": 0.0,
                "power_draw_watts": 50.0,
            },
            {
                "gpu_index": 4,
                "gpu_name": "H100",
                "memory_used_mib": 300.0,
                "utilization_gpu_percent": 90.0,
                "power_draw_watts": 300.0,
            },
            {
                "gpu_index": 5,
                "gpu_name": "H100",
                "memory_used_mib": 120.0,
                "utilization_gpu_percent": 10.0,
                "power_draw_watts": 60.0,
            },
            {
                "gpu_index": 5,
                "gpu_name": "H100",
                "memory_used_mib": 280.0,
                "utilization_gpu_percent": 80.0,
                "power_draw_watts": 280.0,
            },
        ]

        summary = monitor.summary()

        self.assertEqual(summary["max_peak_memory_mib"], 300.0)
        self.assertEqual(summary["sum_per_gpu_peak_memory_mib"], 580.0)
        self.assertEqual(summary["gpus"][0]["peak_memory_delta_mib"], 200.0)

    def test_protocol_fingerprint_ignores_resume_but_not_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = MODULE.build_variant_specs(
                self._checkpoint(root, "full", 256),
                self._checkpoint(root, "p25", 192),
                self._checkpoint(root, "p50", 128),
            )
            settings = {
                "datasets": ["AIME24"],
                "temperature": 1.0,
                "resume": True,
                "run_id": "run-a",
            }
            original = MODULE.build_protocol_fingerprint(settings, specs, "commit-a")
            resumed = MODULE.build_protocol_fingerprint(
                {**settings, "resume": False, "run_id": "run-b"},
                specs,
                "commit-a",
            )
            changed = MODULE.build_protocol_fingerprint(
                {**settings, "temperature": 0.0}, specs, "commit-a"
            )
            changed_code = MODULE.build_protocol_fingerprint(
                settings, specs, "commit-b"
            )

            self.assertEqual(original, resumed)
            self.assertNotEqual(original, changed)
            self.assertNotEqual(original, changed_code)


if __name__ == "__main__":
    unittest.main()
