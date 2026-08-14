#!/usr/bin/env python3
"""Regression tests for the no-download DeepSeek-V4 runtime preflight."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
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
FHT_REPAIR_RESUME = REPO_ROOT / "scripts" / "repair_and_resume_v4_full_reproduction.sh"
GPU_IDLE_CHECK = REPO_ROOT / "scripts" / "check_v4_gpus_idle.sh"
REPRODUCTION_RUNNER = REPO_ROOT / "evaluation" / "run_reproduction_matrix.py"
VENDORED_FHT = REPO_ROOT / "third_party" / "fast-hadamard-transform"


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
        self.assertIn("check_v4_gpus_idle.sh", launcher)
        self.assertIn("os.setsid()", launcher)
        self.assertIn('kill -TERM -- "-${SERVER_PGID}"', launcher)
        self.assertIn('kill -KILL -- "-${SERVER_PGID}"', launcher)

    def test_gpu_idle_gate_reports_busy_process_without_killing_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            self._write_executable(
                fake_bin / "nvidia-smi",
                """\
                #!/usr/bin/env bash
                if [[ "$*" == *"--query-compute-apps="* ]]; then
                  echo '2771319, python, 42840'
                elif [[ "${FAKE_BUSY:-0}" == "1" ]]; then
                  printf '4, 42840\\n5, 42700\\n6, 42720\\n7, 42000\\n'
                else
                  printf '4, 730\\n5, 730\\n6, 730\\n7, 730\\n'
                fi
                """,
            )
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["GPU_LIST"] = "4,5,6,7"

            idle = subprocess.run(
                ["bash", str(GPU_IDLE_CHECK)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(idle.returncode, 0, idle.stdout)
            self.assertIn("GPU exclusivity check: PASS", idle.stdout)

            env["FAKE_BUSY"] = "1"
            busy = subprocess.run(
                ["bash", str(GPU_IDLE_CHECK)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(busy.returncode, 2, busy.stdout)
            self.assertIn("PID 2771319", busy.stdout)
            self.assertIn("no process was terminated", busy.stdout)

    def test_statistics_collector_is_offline_and_restricted_to_gpus_4_7(self) -> None:
        collector = STATISTICS_COLLECTOR.read_text(encoding="utf-8")

        self.assertIn('GPU_LIST="${GPU_LIST:-4,5,6,7}"', collector)
        self.assertIn('[[ "${GPU_LIST}" == "4,5,6,7" ]]', collector)
        self.assertIn("HF_HUB_OFFLINE=1", collector)
        self.assertIn("TRANSFORMERS_OFFLINE=1", collector)
        self.assertIn('"${V4_PYTHON}" -m torch.distributed.run', collector)
        self.assertIn("check_v4_gpus_idle.sh", collector)
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
            'REQUESTED_CONVERTED_CKPT_PATH="${CONVERTED_CKPT_PATH:-${ARTIFACT_ROOT}}"',
            pipeline,
        )
        self.assertIn('CONVERTED_CKPT_PATH="${ARTIFACT_ROOT}"', pipeline)
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
        self.assertIn('ALLOW_DUPLICATE_MP4="${ALLOW_DUPLICATE_MP4:-0}"', pipeline)
        self.assertIn("relocate_mp4_without_copying", pipeline)
        self.assertIn("refusing cross-filesystem relocation", pipeline)
        self.assertIn("os.rename(src, dst)", pipeline)
        self.assertIn("check_v4_gpus_idle.sh", pipeline)
        self.assertIn("Ignored CONVERTED_CKPT_PATH=", pipeline)
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

    def test_full_pipeline_adopts_typo_checkpoint_without_copying(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "v4-converted"
            typo = root / "v4-converte"
            typo.mkdir()
            original_inodes: dict[str, int] = {}
            for rank in range(4):
                shard = typo / f"model{rank}-mp4.safetensors"
                shard.write_bytes(f"rank-{rank}".encode())
                original_inodes[shard.name] = shard.stat().st_ino

            env = os.environ.copy()
            env.update(
                {
                    "ARTIFACT_ROOT": str(canonical),
                    "CONVERTED_CKPT_PATH": str(typo),
                    "RESULTS_ROOT": str(root / "results"),
                    "MASTER_LOG": str(root / "storage-preflight.log"),
                    "V4_PYTHON": sys.executable,
                    "MP4_STORAGE_PREFLIGHT_ONLY": "1",
                }
            )
            result = subprocess.run(
                ["bash", str(FULL_PIPELINE)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            output = result.stdout + (root / "storage-preflight.log").read_text(
                encoding="utf-8"
            )

            self.assertEqual(result.returncode, 0, output)
            self.assertIn("Ignored CONVERTED_CKPT_PATH=", output)
            self.assertIn("renamed without copying", output)
            self.assertIn("no GPU/model operation was run", output)
            for name, inode in original_inodes.items():
                adopted = canonical / name
                self.assertTrue(adopted.is_file(), output)
                self.assertEqual(adopted.stat().st_ino, inode)
                self.assertFalse((typo / name).exists())

    def test_full_pipeline_preflights_official_hadamard_dependency(self) -> None:
        pipeline = FULL_PIPELINE.read_text(encoding="utf-8")
        collector = STATISTICS_COLLECTOR.read_text(encoding="utf-8")

        self.assertIn("from fast_hadamard_transform import hadamard_transform", pipeline)
        self.assertIn("from fast_hadamard_transform import hadamard_transform", collector)
        self.assertIn("repair_and_resume_v4_full_reproduction.sh", pipeline)
        self.assertIn("repair_and_resume_v4_full_reproduction.sh", collector)

    def test_evaluation_runner_checks_gpu_exclusivity_before_server_launch(self) -> None:
        runner = REPRODUCTION_RUNNER.read_text(encoding="utf-8")

        self.assertIn(
            'GPU_IDLE_CHECK = REPO_ROOT / "scripts" / "check_v4_gpus_idle.sh"',
            runner,
        )
        self.assertIn("require_idle_gpus(args.gpu_list)", runner)
        self.assertLess(
            runner.index("require_idle_gpus(args.gpu_list)"),
            runner.index("process = subprocess.Popen(", runner.index("def run_variant(")),
        )

    def test_fht_repair_installs_only_when_cuda_validation_fails(self) -> None:
        repair = FHT_REPAIR_RESUME.read_text(encoding="utf-8")

        self.assertIn('if check_fht >/dev/null 2>&1; then', repair)
        self.assertIn("no download or installation is needed", repair)
        self.assertIn("uv pip install", repair)
        self.assertIn("--offline", repair)
        self.assertIn("--no-build-isolation", repair)
        self.assertIn("--no-deps", repair)
        self.assertIn('"${FHT_SOURCE_DIR}"', repair)
        self.assertIn("FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE", repair)
        self.assertIn("FAST_HADAMARD_TRANSFORM_SKIP_CUDA_BUILD=FALSE", repair)
        self.assertIn("Network package lookup: disabled", repair)
        self.assertIn('[[ "${source_real}" != "/" ]]', repair)
        self.assertIn('rm -rf -- "${build_dir}"', repair)
        self.assertIn("run_v4_full_reproduction_gpus_4_7.sh", repair)
        for forbidden in ("apt install", "apt-get install", "conda install"):
            self.assertNotIn(forbidden, repair)

    def test_vendored_fht_source_is_complete_and_matches_manifest(self) -> None:
        required = (
            "setup.py",
            "LICENSE",
            "fast_hadamard_transform/__init__.py",
            "fast_hadamard_transform/fast_hadamard_transform_interface.py",
            "csrc/fast_hadamard_transform.cpp",
            "csrc/fast_hadamard_transform_cuda.cu",
            "csrc/fast_hadamard_transform.h",
            "csrc/fast_hadamard_transform_common.h",
            "csrc/fast_hadamard_transform_special.h",
            "csrc/static_switch.h",
        )
        for relative in required:
            path = VENDORED_FHT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertGreater(path.stat().st_size, 0, relative)

        manifest = VENDORED_FHT / "SHA256SUMS"
        entries = {}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            entries[relative] = digest
        for relative in required:
            digest = hashlib.sha256((VENDORED_FHT / relative).read_bytes()).hexdigest()
            self.assertEqual(entries.get(relative), digest, relative)

    def test_vendored_fht_avoids_unused_cusparse_and_targets_h100(self) -> None:
        binding = (VENDORED_FHT / "csrc/fast_hadamard_transform.cpp").read_text(
            encoding="utf-8"
        )
        setup = (VENDORED_FHT / "setup.py").read_text(encoding="utf-8")
        repair = FHT_REPAIR_RESUME.read_text(encoding="utf-8")

        self.assertNotIn("ATen/cuda/CUDAContext", binding)
        self.assertIn("c10/cuda/CUDAStream.h", binding)
        self.assertIn("c10::cuda::getCurrentCUDAStream", binding)
        self.assertNotIn("arch=compute_", setup)
        self.assertIn('TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"', repair)


if __name__ == "__main__":
    unittest.main()
