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
              echo '--moe-runner-backend --reasoning-parser --tool-call-parser --disable-custom-all-reduce'
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


if __name__ == "__main__":
    unittest.main()
