#!/usr/bin/env python3
"""Apply/check the EASY-EP mask-routing patch for SGLang v0.5.16.

The patch keeps every DeepSeek-V4 router parameter at 256 rows. For dynamic
layers only, it gathers router logits and correction bias with the persisted
EASY-EP mask before TopK, so compact IDs match the physically retained expert
weights. Hash layers 0..2 and the MTP layer remain unchanged.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import shutil
import sys


EXPECTED_VERSION = "0.5.16"
MARKER = "EASYEP_V4_MASK_ROUTING_V1"
CONFIG_RELATIVE = Path("srt/configs/deepseek_v4.py")
MOE_RELATIVE = Path("srt/models/deepseek_v2.py")

CONFIG_ANCHOR = "    n_routed_experts: int = 256\n"
CONFIG_REPLACEMENT = (
    CONFIG_ANCHOR
    + f"    # {MARKER}: paper-aligned per-layer expert mask.\n"
    + "    easyep_expert_mask_by_layer: Optional[List[List[int]]] = None\n"
    + "    easyep_pruning: Optional[Dict[str, object]] = None\n"
)

MOE_INIT_ANCHOR = """        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts

        n_shared_experts = (
            0 if config.n_shared_experts is None else int(config.n_shared_experts)
        )
        _fusion_disabled = get_server_args().disable_shared_experts_fusion

        # num_fused_shared_experts drives weight remapping in deepseek_weight_loader:
        # mlp.shared_experts → mlp.experts.256 when > 0.
        self.num_fused_shared_experts = 0 if _fusion_disabled else n_shared_experts

        # DeepEP and MegaMOE shared expert fusion: shared expert is fused into
        # the same MoE kernel as a local expert at each EP rank. Expert layout
        # is expanded from 256 routed to 256+EP_size (e.g. 272 for EP=16).
        _uses_per_rank_shared_slots = has_per_rank_fused_shared_slots(
            self.num_fused_shared_experts
        )

        if _uses_per_rank_shared_slots:
            # 256 routed + EP_size shared slots = 272 experts total (for EP=16)
            num_experts_for_moe = config.n_routed_experts + self.moe_ep_size
            top_k_for_moe = config.num_experts_per_tok + 1  # 8 routed + 1 shared
            # Interleaving for DeepEP/MegaMOE dispatch is handled by TopK internally.
        else:
            num_experts_for_moe = (
                config.n_routed_experts + self.num_fused_shared_experts
            )
            top_k_for_moe = config.num_experts_per_tok + self.num_fused_shared_experts

        self.config = config
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.is_nextn = is_nextn

        n_hash_layers = getattr(config, "num_hash_layers", 0)
        self.is_hash = layer_id < n_hash_layers and not (is_deepseek_v4 and is_nextn)

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )
"""

MOE_INIT_REPLACEMENT = f"""        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts

        # {MARKER}: the router remains 256-wide. Only expert storage is compact.
        easyep_masks = getattr(config, "easyep_expert_mask_by_layer", None)
        routed_experts_for_moe = config.n_routed_experts
        selected_expert_ids = None
        if easyep_masks is not None:
            if not is_deepseek_v4:
                raise ValueError("EASY-EP V4 masks cannot be used by a non-V4 model")
            if len(easyep_masks) != config.num_hidden_layers:
                raise ValueError("easyep_expert_mask_by_layer must have one row per layer")
            if not is_nextn:
                row = easyep_masks[layer_id]
                if len(row) != config.n_routed_experts or any(
                    value not in (0, 1, False, True) for value in row
                ):
                    raise ValueError(f"invalid EASY-EP mask at layer {{layer_id}}")
                selected = [index for index, keep in enumerate(row) if int(keep) == 1]
                n_hash_layers = int(getattr(config, "num_hash_layers", 0))
                if layer_id < n_hash_layers:
                    if selected != list(range(config.n_routed_experts)):
                        raise ValueError(f"hash layer {{layer_id}} must retain every expert")
                else:
                    if len(selected) not in (128, 192):
                        raise ValueError(
                            f"dynamic layer {{layer_id}} must retain 128 or 192 experts"
                        )
                    server_args = get_server_args()
                    if not server_args.disable_shared_experts_fusion:
                        raise ValueError(
                            "EASY-EP mask routing requires --disable-shared-experts-fusion"
                        )
                    if server_args.enable_eplb or server_args.ep_num_redundant_experts:
                        raise ValueError(
                            "EASY-EP mask routing requires EPLB and redundant experts disabled"
                        )
                    a2a_backend = get_moe_a2a_backend()
                    if any(
                        check()
                        for check in (
                            a2a_backend.is_deepep,
                            a2a_backend.is_mooncake,
                            a2a_backend.is_nixl,
                            a2a_backend.is_mori,
                            a2a_backend.is_ascend_fuseep,
                            a2a_backend.is_flashinfer,
                            a2a_backend.is_megamoe,
                        )
                    ):
                        raise ValueError(
                            "EASY-EP mask routing currently supports ordinary TP only"
                        )
                    routed_experts_for_moe = len(selected)
                    selected_expert_ids = torch.tensor(selected, dtype=torch.long)

        if selected_expert_ids is None:
            self._easyep_selected_expert_ids = None
        else:
            self.register_buffer(
                "_easyep_selected_expert_ids",
                selected_expert_ids,
                persistent=False,
            )

        n_shared_experts = (
            0 if config.n_shared_experts is None else int(config.n_shared_experts)
        )
        _fusion_disabled = get_server_args().disable_shared_experts_fusion

        # num_fused_shared_experts drives weight remapping in deepseek_weight_loader:
        # mlp.shared_experts → mlp.experts.256 when > 0.
        self.num_fused_shared_experts = 0 if _fusion_disabled else n_shared_experts

        # DeepEP and MegaMOE shared expert fusion: shared expert is fused into
        # the same MoE kernel as a local expert at each EP rank. Expert layout
        # is expanded from 256 routed to 256+EP_size (e.g. 272 for EP=16).
        _uses_per_rank_shared_slots = has_per_rank_fused_shared_slots(
            self.num_fused_shared_experts
        )

        if _uses_per_rank_shared_slots:
            num_experts_for_moe = routed_experts_for_moe + self.moe_ep_size
            top_k_for_moe = config.num_experts_per_tok + 1
        else:
            num_experts_for_moe = (
                routed_experts_for_moe + self.num_fused_shared_experts
            )
            top_k_for_moe = config.num_experts_per_tok + self.num_fused_shared_experts

        self.config = config
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.is_nextn = is_nextn

        n_hash_layers = getattr(config, "num_hash_layers", 0)
        self.is_hash = layer_id < n_hash_layers and not (is_deepseek_v4 and is_nextn)

        if self.tp_size > routed_experts_for_moe:
            raise ValueError(
                f"Tensor parallel size {{self.tp_size}} is greater than "
                f"the number of physical experts {{routed_experts_for_moe}}."
            )
"""

HELPER_ANCHOR = """    def _can_dual_stream_graph(
        self, hidden_states: torch.Tensor, server_args=None
    ) -> bool:
"""
HELPER_REPLACEMENT = f"""    def _apply_easyep_expert_mask(self, router_logits: torch.Tensor):
        # {MARKER}: gather is equivalent to masking unretained experts before
        # TopK, while compact indices match the renumbered physical weights.
        selected = self._easyep_selected_expert_ids
        if selected is None:
            return router_logits
        correction_bias = self.gate.e_score_correction_bias
        if correction_bias is not None:
            self.topk.topk_config.correction_bias = torch.index_select(
                correction_bias, 0, selected
            )
        return torch.index_select(router_logits, -1, selected)

{HELPER_ANCHOR}"""

GATE_PATCHES = [
    (
        "\n        router_logits = self.gate(hidden_states, gemm_output_zero_allocator)\n",
        "\n        router_logits = self._apply_easyep_expert_mask(\n"
        "            self.gate(hidden_states, gemm_output_zero_allocator)\n"
        "        )\n",
        1,
    ),
    (
        "\n            router_logits = self.gate(hidden_states, gemm_output_zero_allocator)\n",
        "\n            router_logits = self._apply_easyep_expert_mask(\n"
        "                self.gate(hidden_states, gemm_output_zero_allocator)\n"
        "            )\n",
        1,
    ),
    (
        "\n        router_logits = self.gate(hidden_states)\n",
        "\n        router_logits = self._apply_easyep_expert_mask(\n"
        "            self.gate(hidden_states)\n"
        "        )\n",
        1,
    ),
    (
        "\n            router_logits = self.gate(hidden_states, forward_batch=forward_batch)\n",
        "\n            router_logits = self._apply_easyep_expert_mask(\n"
        "                self.gate(hidden_states, forward_batch=forward_batch)\n"
        "            )\n",
        1,
    ),
    (
        "\n            state.router_logits = self.gate(state.hidden_states_mlp_input)\n",
        "\n            state.router_logits = self._apply_easyep_expert_mask(\n"
        "                self.gate(state.hidden_states_mlp_input)\n"
        "            )\n",
        1,
    ),
]


def patch_specs(package_root: Path):
    return {
        package_root / CONFIG_RELATIVE: [
            (CONFIG_ANCHOR, CONFIG_REPLACEMENT, 1),
        ],
        package_root / MOE_RELATIVE: [
            (MOE_INIT_ANCHOR, MOE_INIT_REPLACEMENT, 1),
            (HELPER_ANCHOR, HELPER_REPLACEMENT, 1),
            *GATE_PATCHES,
        ],
    }


def discover_package_root() -> Path:
    version = importlib.metadata.version("sglang")
    if version != EXPECTED_VERSION:
        raise RuntimeError(
            f"expected SGLang {EXPECTED_VERSION}, found {version}; refusing to patch"
        )
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("cannot locate the installed sglang package")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def compile_source(path: Path, source: str) -> None:
    compile(source, str(path), "exec")


def atomic_write(path: Path, source: str) -> None:
    temporary = path.with_suffix(path.suffix + ".easyep.tmp")
    temporary.write_text(source, encoding="utf-8")
    os.replace(temporary, path)


def backup_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + f".easyep-v4-mask-{EXPECTED_VERSION}.bak")


def patched_source(path: Path, source: str, specs) -> str:
    updated = source
    for anchor, replacement, expected_count in specs:
        count = updated.count(anchor)
        if count != expected_count:
            raise RuntimeError(
                f"expected {expected_count} patch anchor(s) in {path}, found {count}; "
                f"installed source differs from supported SGLang {EXPECTED_VERSION}"
            )
        updated = updated.replace(anchor, replacement, expected_count)
    compile_source(path, updated)
    return updated


def check(package_root: Path) -> None:
    for path, specs in patch_specs(package_root).items():
        if not path.is_file():
            raise FileNotFoundError(f"SGLang source file is missing: {path}")
        source = path.read_text(encoding="utf-8")
        if MARKER not in source or any(replacement not in source for _, replacement, _ in specs):
            raise RuntimeError(f"EASY-EP mask-routing patch is incomplete in {path}")
        compile_source(path, source)
    print(f"SGLang {EXPECTED_VERSION} EASY-EP mask-routing patch verified: {package_root}")


def apply(package_root: Path) -> None:
    planned: list[tuple[Path, str]] = []
    for path, specs in patch_specs(package_root).items():
        if not path.is_file():
            raise FileNotFoundError(f"SGLang source file is missing: {path}")
        source = path.read_text(encoding="utf-8")
        if MARKER in source:
            if any(replacement not in source for _, replacement, _ in specs):
                raise RuntimeError(f"EASY-EP mask-routing patch is incomplete in {path}")
            compile_source(path, source)
            print(f"already patched: {path}")
            continue
        planned.append((path, patched_source(path, source, specs)))

    for path, source in planned:
        backup = backup_path(path)
        if not backup.exists():
            shutil.copy2(path, backup)
        atomic_write(path, source)
        print(f"patched: {path}; backup={backup}")
    check(package_root)
    if planned:
        print("restart SGLang processes so the patched modules are imported")


def restore(package_root: Path) -> None:
    restored = 0
    for path in patch_specs(package_root):
        backup = backup_path(path)
        if not backup.is_file():
            raise FileNotFoundError(f"patch backup is missing: {backup}")
        source = backup.read_text(encoding="utf-8")
        compile_source(path, source)
        atomic_write(path, source)
        restored += 1
        print(f"restored: {path}")
    print(f"restored {restored} SGLang source files; restart SGLang processes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Patch SGLang 0.5.16 for EASY-EP V4 mask routing"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--apply", action="store_true")
    action.add_argument("--restore", action="store_true")
    parser.add_argument(
        "--package-root",
        type=Path,
        help="test/debug override; defaults to the active interpreter's sglang package",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    package_root = (
        args.package_root.expanduser().resolve()
        if args.package_root is not None
        else discover_package_root()
    )
    if args.apply:
        apply(package_root)
    elif args.restore:
        restore(package_root)
    else:
        check(package_root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
