#!/usr/bin/env python3
"""Resolve discrete DeepSeek-V4 expert counts from requested prune rates."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path
from typing import Iterable


def parse_percent(value: str | int | float | Decimal) -> Decimal:
    """Parse a percentage in ``[0, 100)`` without binary-float rounding."""

    try:
        percent = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"invalid prune percentage: {value!r}") from exc
    if not percent.is_finite() or percent < 0 or percent >= 100:
        raise ValueError(f"prune percentage must be in [0, 100), found {value!r}")
    return percent


def target_from_prune_percent(
    prune_percent: str | int | float | Decimal,
    num_experts: int,
) -> int:
    """Return a retained count that never prunes more than requested.

    Expert counts are discrete.  We round the retained count upward, so a
    request such as 30% on 256 experts keeps 180 experts and actually prunes
    29.6875%, rather than silently exceeding the requested rate.
    """

    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, found {num_experts}")
    percent = parse_percent(prune_percent)
    retained = (
        Decimal(num_experts) * (Decimal(100) - percent) / Decimal(100)
    ).to_integral_value(rounding=ROUND_CEILING)
    return int(retained)


def actual_prune_percent(target_experts: int, num_experts: int) -> Decimal:
    if not 1 <= target_experts <= num_experts:
        raise ValueError(
            f"target_experts must be in [1, {num_experts}], found {target_experts}"
        )
    return (
        Decimal(num_experts - target_experts) * Decimal(100) / Decimal(num_experts)
    )


def resolve_targets(
    target_experts: Iterable[int],
    prune_percentages: Iterable[str | int | float | Decimal],
    num_experts: int,
) -> list[int]:
    """Resolve, validate, de-duplicate, and preserve requested target order."""

    resolved = [int(value) for value in target_experts]
    resolved.extend(
        target_from_prune_percent(value, num_experts) for value in prune_percentages
    )
    if not resolved:
        raise ValueError("provide at least one target expert count or prune percentage")

    unique: list[int] = []
    seen: set[int] = set()
    for target in resolved:
        if not 1 <= target <= num_experts:
            raise ValueError(
                f"target_experts must be in [1, {num_experts}], found {target}"
            )
        if target not in seen:
            seen.add(target)
            unique.append(target)
    return unique


def decimal_text(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return "0" if text == "-0" else text


def percent_slug(value: Decimal) -> str:
    return decimal_text(value).replace(".", "p")


def mask_filename(prefix: str, target_experts: int, num_experts: int) -> str:
    percent = percent_slug(actual_prune_percent(target_experts, num_experts))
    return f"{prefix}_prune{percent}_keep{target_experts}.json"


def model_directory_name(target_experts: int, num_experts: int) -> str:
    percent = percent_slug(actual_prune_percent(target_experts, num_experts))
    return f"v4-prune{percent}-keep{target_experts}"


def validate_safe_prefix(value: str) -> str:
    if not value or value in {".", ".."}:
        raise ValueError("mask prefix must not be empty or a path traversal component")
    path = Path(value)
    allowed = "-_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    if path.name != value or any(character not in allowed for character in value):
        raise ValueError(
            "mask prefix must contain only letters, digits, dot, dash, or underscore"
        )
    return value
