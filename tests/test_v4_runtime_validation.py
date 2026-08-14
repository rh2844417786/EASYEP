#!/usr/bin/env python3
"""Regression tests for the no-download DeepSeek-V4 runtime preflight."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "scripts" / "validate_v4_flash_runtime.sh"
FOUR_GPU_LAUNCHER = REPO_ROOT / "scripts" / "test_v4_flash_gpus_4_7.sh"
STATISTICS_COLLECTOR = (
    REPO_ROOT / "scripts" / "collect_v4_easyep_statistics_gpus_4_7.sh"
)
PRUNED_PREPARER = REPO_ROOT / "scripts" / "prepare_v4_pruned_checkpoints.sh"
FULL_PIPELINE = REPO_ROOT / "scripts" / "run_v4_full_reproduction_gpus_4_7.sh"


class V4RuntimeValidationTests(unittest.TestCase):
    def _write_executable(self, path: Path, body: str) -> None:
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        path.chmod(0o755)

    def _fixture(self, root: Path) -> tuple[dict[str, str], Path]:
        fake_bin = root / "bin"
        fake_bin.mkdir()
        marker = root / "unexpected-download-command"
        model = root / "model"
        model.mkdir()

        self._write_executable(
            fake_bin / "ps",
            """\
            #!/usr/bin/env bash
            if [[ "${FAKE_ACTIVE_TRANSFER:-0}" == "1" ]]; then
              echo "4321 1 00:12 curl"
            fi
            """,
        )
        self._write_executable(
            fake_bin / "nvidia-smi",
            """\
            #!/usr/bin/env bash
            cat <<'EOF'
            4, NVIDIA H100 80GB HBM3, 80000
            5, NVIDIA H100 80GB HBM3, 80000
            6, NVIDIA H100 80GB HBM3, 80000
            7, NVIDIA H100 80GB HBM3, 80000
            EOF
            """,
        )
        self._write_executable(
            fake_bin / "nvcc",
            """\
            #!/usr/bin/env bash
            echo 'Cuda compilation tools, release 13.0, V13.0.88'
            """,
        )
        self._write_executable(
            fake_bin / "v4-python",
            """\
            #!/usr/bin/env bash
            if [[ "${1:-}" == "-m" ]]; then
              echo '--moe-runner-backend --reasoning-parser --tool-call-parser --watchdog-timeout --disable-custom-all-reduce --disable-shared-experts-fusion'
            elif [[ "${1:-}" == "-c" ]]; then
              echo '/opt/sglang-v4/lib/python3.11/site-packages/sglang/__init__.py'
            else
              echo 'mock runtime/model validation passed'
            fi
            """,
        )
        for name in ("apt", "apt-get", "dpkg", "curl", "wget", "pip", "pip3", "uv"):
            self._write_executable(
                fake_bin / name,
                f"""\
                #!/usr/bin/env bash
                echo {name} >>"${{DOWNLOAD_MARKER}}"
                exit 97
                """,
            )

        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{fake_bin}:{env['PATH']}",
                "DG_JIT_NVCC_COMPILER": str(fake_bin / "nvcc"),
                "V4_PYTHON": str(fake_bin / "v4-python"),
                "MODEL_PATH": str(model),
                "DOWNLOAD_MARKER": str(marker),
            }
        )
        return env, marker

    def test_success_path_never_calls_downloader_or_installer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, marker = self._fixture(Path(tmp))
            result = subprocess.run(
                ["bash", str(VALIDATOR)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("Validation passed", result.stdout)
            self.assertIn("Hugging Face/Transformers offline mode: enabled", result.stdout)
            self.assertFalse(marker.exists(), result.stdout)

    def test_active_transfer_aborts_before_runtime_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env, marker = self._fixture(Path(tmp))
            env["FAKE_ACTIVE_TRANSFER"] = "1"
            result = subprocess.run(
                ["bash", str(VALIDATOR)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("Active transfer/install process", result.stdout)
            self.assertIn("Refusing to overlap", result.stdout)
            self.assertFalse(marker.exists(), result.stdout)

    def test_four_gpu_launcher_extends_first_jit_timeouts(self) -> None:
        launcher = FOUR_GPU_LAUNCHER.read_text(encoding="utf-8")

        self.assertIn('WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-1800}"', launcher)
        self.assertIn('SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-1800}"', launcher)
        self.assertIn('--watchdog-timeout "${WATCHDOG_TIMEOUT}"', launcher)
        self.assertIn('--disable-shared-experts-fusion', launcher)
        self.assertIn('--timeout "${SMOKE_TIMEOUT}"', launcher)

    def test_statistics_collector_is_offline_and_restricted_to_gpus_4_7(self) -> None:
        collector = STATISTICS_COLLECTOR.read_text(encoding="utf-8")

        self.assertIn('GPU_LIST="${GPU_LIST:-4,5,6,7}"', collector)
        self.assertIn('[[ "${GPU_LIST}" == "4,5,6,7" ]]', collector)
        self.assertIn("HF_HUB_OFFLINE=1", collector)
        self.assertIn("TRANSFORMERS_OFFLINE=1", collector)
        self.assertIn('"${V4_PYTHON}" -m torch.distributed.run', collector)
        for forbidden in ("apt install", "pip install", "uv pip", "curl ", "wget "):
            self.assertNotIn(forbidden, collector)

    def test_pruned_preparer_auto_collects_only_when_statistics_are_missing(self) -> None:
        preparer = PRUNED_PREPARER.read_text(encoding="utf-8")

        self.assertIn('if [[ ! -f "${TOKEN_STATS}" ]]', preparer)
        self.assertIn("collect_v4_easyep_statistics_gpus_4_7.sh", preparer)
        self.assertIn("V4 token statistics already exist", preparer)

    def test_full_pipeline_uses_external_artifact_root_and_is_fail_fast(self) -> None:
        pipeline = FULL_PIPELINE.read_text(encoding="utf-8")

        self.assertIn("set -Eeuo pipefail", pipeline)
        self.assertIn(
            'ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/docker_data/v4-converted}"',
            pipeline,
        )
        self.assertIn(
            'CONVERTED_CKPT_PATH="${CONVERTED_CKPT_PATH:-${ARTIFACT_ROOT}/mp4-fp4}"',
            pipeline,
        )
        self.assertIn(
            'PRUNE25_MODEL_PATH="${PRUNE25_MODEL_PATH:-${ARTIFACT_ROOT}/v4-prune25-keep192}"',
            pipeline,
        )
        self.assertIn(
            'PRUNE50_MODEL_PATH="${PRUNE50_MODEL_PATH:-${ARTIFACT_ROOT}/v4-prune50-keep128}"',
            pipeline,
        )
        self.assertIn("--model-parallel 4", pipeline)
        self.assertIn("--expert-dtype fp4", pipeline)
        self.assertIn("validate_mp4", pipeline)
        self.assertIn(
            'DRY_RUN=1 bash "${SCRIPT_DIR}/run_easyep_reproduction.sh"',
            pipeline,
        )
        self.assertIn(
            'DRY_RUN=0 bash "${SCRIPT_DIR}/run_easyep_reproduction.sh"',
            pipeline,
        )

    def test_full_pipeline_does_not_download_or_install(self) -> None:
        pipeline = FULL_PIPELINE.read_text(encoding="utf-8")

        for forbidden in (
            "apt install",
            "apt-get install",
            "pip install",
            "uv pip",
            "curl ",
            "wget ",
            "huggingface-cli download",
        ):
            self.assertNotIn(forbidden, pipeline)


if __name__ == "__main__":
    unittest.main()
