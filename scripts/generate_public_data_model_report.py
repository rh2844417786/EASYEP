#!/usr/bin/env python3
"""Generate a checked-in inventory of Hugging Face model roots in public_data."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


def is_model_config(config_path: Path) -> bool:
    try:
        content = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return "model_type" in content or "architectures" in content


def discover_model_roots(root: Path) -> list[Path]:
    return sorted(
        config_path.parent
        for config_path in root.rglob("config.json")
        if is_model_config(config_path)
    )


def render_report(model_roots: list[Path], output: Path) -> None:
    lines = [
        "# Public Data Model Inventory",
        "",
        f"Snapshot date: {date.today().isoformat()}",
        "",
        "This report lists every detected Hugging Face model root under "
        "`/mnt/public_data`. A directory is included when its `config.json` "
        "is valid JSON and contains either `model_type` or `architectures`. "
        "This excludes datasets, source repositories, caches, and model "
        "subdirectories without their own model configuration.",
        "",
        f"Detected model roots: {len(model_roots)}",
        "",
        "Each entry includes the exact `config.json` read at report generation time.",
        "",
    ]

    for index, model_root in enumerate(model_roots, start=1):
        config_path = model_root / "config.json"
        config_text = config_path.read_text(encoding="utf-8").rstrip()
        lines.extend(
            [
                f"## {index}. `{model_root}`",
                "",
                "```json",
                config_text,
                "```",
                "",
            ]
        )

    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/public_data"))
    parser.add_argument(
        "--output", type=Path, default=Path("PUBLIC_DATA_MODEL_REPORT.md")
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional newline-delimited model-root list from a completed scan.",
    )
    args = parser.parse_args()

    if args.input:
        model_roots = sorted(
            {
                (Path(line.strip()).parent if Path(line.strip()).name == "config.json" else Path(line.strip()))
                for line in args.input.read_text(encoding="utf-8").splitlines()
                if line.strip()
                and is_model_config(
                    Path(line.strip())
                    if Path(line.strip()).name == "config.json"
                    else Path(line.strip()) / "config.json"
                )
            }
        )
    else:
        model_roots = discover_model_roots(args.root)

    render_report(model_roots, args.output)


if __name__ == "__main__":
    main()
