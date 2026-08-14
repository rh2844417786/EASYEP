from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import v4_environment_report


class V4EnvironmentReportTests(unittest.TestCase):
    def test_sanitize_redacts_tokens_and_url_credentials(self):
        text = (
            "https://alice:secret@example.com/repo "
            "github_pat_abcdefghijklmnopqrstuvwxyz123456 "
            "gho_abcdefghijklmnopqrstuvwxyz123456"
        )
        redacted = v4_environment_report.sanitize(text)
        self.assertNotIn("secret", redacted)
        self.assertNotIn("github_pat_", redacted)
        self.assertNotIn("gho_", redacted)
        self.assertIn("<redacted", redacted)

    def test_parse_gpu_inventory_accepts_h100_rows(self):
        fields = ["index", "name", "memory.total", "memory.free"]
        inventory = v4_environment_report.parse_gpu_inventory(
            "4, NVIDIA H100 80GB HBM3, 81559, 81500\n"
            "5, NVIDIA H100 80GB HBM3, 81559, 81400\n",
            fields,
        )
        self.assertEqual([row["index"] for row in inventory], ["4", "5"])
        self.assertEqual(inventory[0]["memory.free"], "81500")

    def test_parse_gpu_ids_rejects_duplicates(self):
        with self.assertRaises(Exception):
            v4_environment_report.parse_gpu_ids("4,4")

    def test_version_tuple_accepts_release_suffix(self):
        self.assertEqual(v4_environment_report.version_tuple("0.5.16.post1"), (0, 5, 16))

    def test_gpu_collector_accepts_indexes_four_through_seven(self):
        inventory = "\n".join(
            f"{index}, NVIDIA H100 80GB HBM3, GPU-{index}, 580.173.02, 9.0, 81559, 0, 81559"
            for index in range(4, 8)
        )

        def fake_run(command, **_kwargs):
            if any(str(argument).startswith("--query-gpu=") for argument in command):
                return v4_environment_report.CommandResult(command, 0, inventory, "")
            return v4_environment_report.CommandResult(command, 0, "ok", "")

        report = v4_environment_report.Report(model_path=None, gpu_ids=[4, 5, 6, 7])
        with mock.patch.object(
            v4_environment_report.shutil,
            "which",
            side_effect=lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
        ), mock.patch.object(v4_environment_report, "run_command", side_effect=fake_run):
            v4_environment_report.collect_gpu_environment(
                report,
                [4, 5, 6, 7],
                timeout=5,
                skip_cuda_smoke=True,
            )

        findings = {(finding.status, finding.check) for finding in report.findings}
        self.assertIn(("PASS", "Requested GPU visibility"), findings)
        self.assertIn(("PASS", "Requested GPU free memory"), findings)

    def test_embedded_torch_smoke_is_valid_python(self):
        compile(v4_environment_report.TORCH_CUDA_SMOKE, "<torch-smoke>", "exec")

    def test_report_renders_model_findings_without_full_config(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "config.json").write_text(
                json.dumps({
                    "model_type": "deepseek_v4",
                    "expert_dtype": "fp4",
                    "hidden_size": 4096,
                    "private_field": "must-not-appear",
                }),
                encoding="utf-8",
            )
            (model_path / "model-00001-of-00001.safetensors").write_bytes(b"1234")
            (model_path / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": 4}, "weight_map": {"w": "model-00001-of-00001.safetensors"}}),
                encoding="utf-8",
            )
            report = v4_environment_report.Report(model_path=model_path, gpu_ids=[4, 5, 6, 7])
            v4_environment_report.collect_model(report, model_path, timeout=5)
            rendered = report.render()
            self.assertIn("deepseek_v4", rendered)
            self.assertIn("fp4", rendered)
            self.assertNotIn("must-not-appear", rendered)
            self.assertNotIn("private_field", rendered)


if __name__ == "__main__":
    unittest.main()
