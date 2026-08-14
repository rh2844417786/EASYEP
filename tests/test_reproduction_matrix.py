#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock


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
        hash_layers: int = 3,
        marker: str | None = None,
    ) -> Path:
        path = root / name
        path.mkdir()
        config = {
            "model_type": "deepseek_v4",
            "n_routed_experts": 256,
            "num_hash_layers": hash_layers,
            "num_hidden_layers": 43,
            "test_marker": marker if marker is not None else name,
        }
        if experts != 256:
            plan_fingerprint = hashlib.sha256(name.encode()).hexdigest()
            dynamic_row = [1] * experts + [0] * (256 - experts)
            config["easyep_expert_mask_by_layer"] = (
                [[1] * 256 for _ in range(hash_layers)]
                + [list(dynamic_row) for _ in range(43 - hash_layers)]
            )
            config["easyep_pruning"] = {
                "format_version": 1,
                "scope": "dynamic_moe_layers_only",
                "hash_layers_preserved": True,
                "hash_layer_ids": list(range(hash_layers)),
                "dynamic_layer_ids": list(range(hash_layers, 43)),
                "original_experts_per_layer": 256,
                "target_dynamic_experts_per_layer": experts,
                "dynamic_layer_prune_fraction": 1 - experts / 256,
                "main_layer_slot_prune_fraction": (
                    1 - (hash_layers * 256 + (43 - hash_layers) * experts) / (43 * 256)
                ),
                "mask_sha256": "a" * 64,
                "plan_fingerprint": plan_fingerprint,
                "router_parameters_pruned": False,
                "router_mask_applied_at_runtime": True,
                "mtp_pruned": False,
            }
            (path / "easyep_pruning_manifest.json").write_text(
                json.dumps(
                    {
                        "plan_fingerprint": plan_fingerprint,
                        "router_parameters_pruned": False,
                        "router_mask_applied_at_runtime": True,
                        "layout": {
                            "counts_by_layer": (
                                [256] * hash_layers + [experts] * (43 - hash_layers)
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (path / "model-00001-of-00001.safetensors").write_bytes(name.encode())
        counts = [256] * hash_layers + [experts] * (43 - hash_layers)
        weight_map = {
            f"model.layers.{layer}.mlp.experts.{expert}.w1.weight": (
                "model-00001-of-00001.safetensors"
            )
            for layer, count in enumerate(counts)
            for expert in range(count)
        }
        for layer in range(43):
            weight_map[f"model.layers.{layer}.mlp.gate.weight"] = (
                "model-00001-of-00001.safetensors"
            )
            gate_field = "tid2eid" if layer < hash_layers else "bias"
            weight_map[f"model.layers.{layer}.mlp.gate.{gate_field}"] = (
                "model-00001-of-00001.safetensors"
            )
        index = {
            "metadata": {},
            "weight_map": weight_map,
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

    def test_hash_layers_are_preserved_while_dynamic_layers_are_pruned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = self._checkpoint(root, "full", 256, hash_layers=3)
            p25 = self._checkpoint(root, "p25", 192, hash_layers=3)
            p50 = self._checkpoint(root, "p50", 128, hash_layers=3)
            specs = MODULE.build_variant_specs(full, p25, p50)

            self.assertEqual(specs[1].checkpoint.hash_routed_experts, 256)
            self.assertEqual(specs[1].checkpoint.routed_experts, 192)
            self.assertAlmostEqual(
                specs[2].checkpoint.main_layer_slot_prune_fraction,
                1 - (3 * 256 + 40 * 128) / (43 * 256),
            )

            bad_config_path = p25 / "config.json"
            bad_config = json.loads(bad_config_path.read_text(encoding="utf-8"))
            bad_config["easyep_expert_mask_by_layer"][0][0] = 0
            bad_config_path.write_text(json.dumps(bad_config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prunes a hash layer"):
                MODULE.build_variant_specs(full, p25, p50)

    def test_expected_expert_counts_produce_distinct_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = self._checkpoint(root, "full", 256, marker="same")
            p25 = self._checkpoint(root, "p25", 192, marker="same")
            p50 = self._checkpoint(root, "p50", 128, marker="same")
            # Different expert counts make the config fingerprints distinct.
            specs = MODULE.build_variant_specs(full, p25, p50)
            self.assertEqual(len({spec.checkpoint.fingerprint for spec in specs}), 3)

    def test_config_only_pruning_is_rejected_by_index_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = self._checkpoint(root, "full", 256)
            p25 = self._checkpoint(root, "p25", 192)
            p50 = self._checkpoint(root, "p50", 128)
            full_index = (full / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
            (p25 / "model.safetensors.index.json").write_text(
                full_index, encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "expert ids do not match config"):
                MODULE.build_variant_specs(full, p25, p50)

    def test_parse_evaluation_output_preserves_accuracy_time_and_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "AIME24.jsonl"
            records = [
                {
                    "type": "sample",
                    "job_id": "test:0",
                    "latency_seconds": 1.0,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
                {
                    "type": "sample",
                    "job_id": "test:1",
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

    def test_gpu_monitor_takes_synchronous_prelaunch_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls = 0

            def fake_nvidia_smi(*_args, **_kwargs):
                nonlocal calls
                memory = 100 if calls == 0 else 300
                calls += 1
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        f"4, H100, {memory}, 81079, 0, 50\n"
                        f"5, H100, {memory}, 81079, 0, 50\n"
                    ),
                    stderr="",
                )

            monitor = MODULE.GpuMonitor(
                Path(tmp) / "gpu.csv", "full", "4,5", 0.01
            )
            with mock.patch.object(
                MODULE.subprocess, "run", side_effect=fake_nvidia_smi
            ):
                monitor.start()
                time.sleep(0.03)
                monitor.stop()

            summary = monitor.summary()
            self.assertEqual(summary["gpus"][0]["baseline_memory_mib"], 100.0)
            self.assertEqual(summary["gpus"][0]["peak_memory_mib"], 300.0)
            trace = (Path(tmp) / "gpu.csv").read_text(encoding="utf-8")
            self.assertIn(",baseline,", trace)
            self.assertIn(",runtime,", trace)

    def test_gpu_monitor_rejects_incomplete_prelaunch_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            monitor = MODULE.GpuMonitor(
                Path(tmp) / "gpu.csv", "full", "4,5", 1.0
            )
            incomplete = SimpleNamespace(
                returncode=0,
                stdout="4, H100, 100, 81079, 0, 50\n",
                stderr="",
            )
            with mock.patch.object(MODULE.subprocess, "run", return_value=incomplete):
                with self.assertRaisesRegex(RuntimeError, "each requested GPU"):
                    monitor.start()

    def test_checkpoint_fingerprint_includes_weight_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._checkpoint(root, "full", 256)
            first = MODULE.inspect_checkpoint(checkpoint)
            shard = checkpoint / "model-00001-of-00001.safetensors"
            shard.write_bytes(b"different-weight-payload")
            second = MODULE.inspect_checkpoint(checkpoint)

            self.assertNotEqual(first.fingerprint, second.fingerprint)
            self.assertNotEqual(first.shard_sha256, second.shard_sha256)

    def test_cached_dataset_requires_matching_raw_jsonl_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = MODULE.inspect_checkpoint(
                self._checkpoint(root, "full", 256)
            )
            spec = MODULE.VariantSpec("full", 0.0, 256, checkpoint)
            variant_dir = root / "run" / "full"
            output = variant_dir / "evaluation" / "AIME24.jsonl"
            output.parent.mkdir(parents=True)
            records = [
                {
                    "type": "sample",
                    "job_id": "cache:0",
                    "latency_seconds": 1.0,
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                },
                {
                    "type": "summary",
                    "accuracy": 100.0,
                    "correct": 1,
                    "total": 1,
                    "dataset": "AIME24",
                    "model": "test-model",
                    "run_fingerprint": "cache-run",
                    "evaluator_backend": "test",
                },
            ]
            output.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            parsed = MODULE.parse_evaluation_output(output, 4.0)
            previous = {
                "protocol_fingerprint": "protocol-a",
                "checkpoint": {"fingerprint": checkpoint.fingerprint},
                "served_model": "test-model",
                "datasets": {"AIME24": parsed},
            }
            args = SimpleNamespace(
                protocol_fingerprint="protocol-a",
                datasets=["AIME24"],
                dataset_totals={"AIME24": 1},
                repeats=1,
            )

            valid, reasons = MODULE.validated_cached_datasets(
                previous, args, spec, variant_dir
            )
            self.assertEqual(reasons, [])
            self.assertIn("AIME24", valid)

            output.write_text(output.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            valid, reasons = MODULE.validated_cached_datasets(
                previous, args, spec, variant_dir
            )
            self.assertEqual(valid, {})
            self.assertTrue(any("hash" in reason for reason in reasons), reasons)

    def test_parse_evaluation_output_rejects_truncated_but_self_consistent_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "AIME24.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "type": "sample",
                        "job_id": "truncated:0",
                        "latency_seconds": 1.0,
                        "usage": {},
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "type": "summary",
                        "accuracy": 100.0,
                        "correct": 1,
                        "total": 1,
                        "dataset": "AIME24",
                        "model": "test-model",
                        "run_fingerprint": "truncated",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "expected 2"):
                MODULE.parse_evaluation_output(
                    output,
                    1.0,
                    expected_dataset="AIME24",
                    expected_model="test-model",
                    expected_total=2,
                )

    def test_completed_variant_requires_valid_telemetry_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            variant_dir = Path(tmp)
            telemetry = variant_dir / "gpu_telemetry.csv"
            telemetry.write_text(
                "timestamp_utc,epoch_seconds,phase,variant,gpu_index,gpu_name,"
                "memory_used_mib,memory_total_mib,utilization_gpu_percent,power_draw_watts\n"
                "now,1,baseline,full,4,H100,100,81079,0,50\n"
                "now,2,runtime,full,4,H100,300,81079,90,300\n",
                encoding="utf-8",
            )
            server_log = variant_dir / "server.log"
            smoke_log = variant_dir / "smoke.log"
            server_log.write_text("server ready\n", encoding="utf-8")
            smoke_log.write_text("READY\n", encoding="utf-8")
            gpu_summary = MODULE.summarize_gpu_samples(
                [
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
                ],
                {4: 100.0},
            )
            previous = {
                "variant": "full",
                "gpu": {
                    **gpu_summary,
                    "errors": [],
                    "trace_bytes": telemetry.stat().st_size,
                    "trace_sha256": MODULE.sha256_file(telemetry),
                },
                "artifacts": {
                    "server_log": MODULE.file_evidence(server_log),
                    "smoke_log": MODULE.file_evidence(smoke_log),
                },
            }

            self.assertEqual(
                MODULE.validate_completed_variant_artifacts(
                    previous, variant_dir, "4"
                ),
                [],
            )
            telemetry.write_text("corrupt\n", encoding="utf-8")
            reasons = MODULE.validate_completed_variant_artifacts(
                previous, variant_dir, "4"
            )
            self.assertTrue(any("telemetry" in reason for reason in reasons), reasons)

    def test_completed_variant_rejects_inconsistent_gpu_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            variant_dir = Path(tmp)
            telemetry = variant_dir / "gpu_telemetry.csv"
            telemetry.write_text(
                "timestamp_utc,epoch_seconds,phase,variant,gpu_index,gpu_name,"
                "memory_used_mib,memory_total_mib,utilization_gpu_percent,power_draw_watts\n"
                "now,1,baseline,full,4,H100,100,81079,0,50\n"
                "now,2,runtime,full,4,H100,300,81079,90,300\n",
                encoding="utf-8",
            )
            server_log = variant_dir / "server.log"
            smoke_log = variant_dir / "smoke.log"
            server_log.write_text("server ready\n", encoding="utf-8")
            smoke_log.write_text("READY\n", encoding="utf-8")
            previous = {
                "variant": "full",
                "gpu": {
                    "errors": [],
                    "sample_rows": 2,
                    "gpus": [],
                    "max_peak_memory_mib": 1.0,
                    "sum_per_gpu_peak_memory_mib": 1.0,
                    "trace_bytes": telemetry.stat().st_size,
                    "trace_sha256": MODULE.sha256_file(telemetry),
                },
                "artifacts": {
                    "server_log": MODULE.file_evidence(server_log),
                    "smoke_log": MODULE.file_evidence(smoke_log),
                },
            }

            reasons = MODULE.validate_completed_variant_artifacts(
                previous, variant_dir, "4"
            )
            self.assertTrue(any("GPU summary" in reason for reason in reasons), reasons)

    @unittest.skipUnless(hasattr(os, "killpg"), "requires POSIX process groups")
    def test_stop_process_group_kills_child_after_leader_has_exited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready = root / "ready"
            terminated = root / "terminated"
            child_code = (
                "import pathlib,signal,sys,time; "
                "ready=pathlib.Path(sys.argv[1]); done=pathlib.Path(sys.argv[2]); "
                "signal.signal(signal.SIGTERM, lambda *_: "
                "(done.write_text('term'), sys.exit(0))); "
                "ready.write_text('ready'); "
                "time.sleep(300)"
            )
            # Use a short parent script with an explicit wait loop; the spawned
            # child remains in the parent's new process group after leader exit.
            parent_code = "\n".join(
                [
                    "import pathlib, subprocess, sys, time",
                    f"ready = pathlib.Path({str(ready)!r})",
                    f"child_code = {child_code!r}",
                    (
                        "child = subprocess.Popen([sys.executable, '-c', child_code, "
                        f"{str(ready)!r}, {str(terminated)!r}], "
                        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"
                    ),
                    "deadline = time.time() + 5",
                    "while not ready.exists() and time.time() < deadline: time.sleep(0.05)",
                    "print(child.pid, flush=True)",
                ]
            )
            process = subprocess.Popen(
                [sys.executable, "-c", parent_code],
                text=True,
                stdout=subprocess.PIPE,
                start_new_session=True,
            )
            assert process.stdout is not None
            child_pid = int(process.stdout.readline().strip())
            process.stdout.close()
            process.wait(timeout=5)
            try:
                MODULE.stop_process_group(process, timeout=5)
                deadline = time.monotonic() + 5
                while not terminated.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(terminated.exists())
            finally:
                try:
                    os.kill(child_pid, 9)
                except ProcessLookupError:
                    pass

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
            changed_runtime = MODULE.build_protocol_fingerprint(
                {**settings, "runtime_identity": {"sglang": "different"}},
                specs,
                "commit-a",
            )

            self.assertEqual(original, resumed)
            self.assertNotEqual(original, changed)
            self.assertNotEqual(original, changed_code)
            self.assertNotEqual(original, changed_runtime)

    def test_wrapper_defaults_to_repository_models_directory(self) -> None:
        wrapper = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_easyep_reproduction.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("models/v4-prune25-keep192", wrapper)
        self.assertIn("models/v4-prune50-keep128", wrapper)


if __name__ == "__main__":
    unittest.main()
