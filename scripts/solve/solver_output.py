#!/usr/bin/env python3
"""Strict parser for the production solver's machine-readable result tokens."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

_RESULT_KEYS = ("n", "residue", "modulus", "wall_s")
_RESULT_MARKER_KEYS = ("residue", "modulus", "wall_s")
_UINT_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
# Matches the forms emitted by C++ iostreams for finite nonnegative doubles,
# while rejecting signs on the mantissa, NaN/Inf, and non-canonical integers.
_FLOAT_RE = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


@dataclass(frozen=True)
class SolverResult:
    n: int
    residue: int
    modulus: int
    wall_s: float


def parse_result_line(line: str) -> SolverResult | None:
    """Parse one stdout line.

    Diagnostic lines without result markers return ``None``. If any result
    marker is present, all four required tokens (n/residue/modulus/wall_s) must occur exactly once and
    have canonical values. Other ``key=value`` diagnostic tokens are allowed.
    """
    values: dict[str, str] = {}
    for token in line.strip().split():
        for key in _RESULT_KEYS:
            prefix = f"{key}="
            if not token.startswith(prefix):
                continue
            if key in values:
                raise ValueError(f"duplicate solver result field: {key}")
            values[key] = token[len(prefix):]
            break

    # A standalone n=... token may legitimately appear in diagnostics.  Only
    # the result-specific markers make a line a candidate result record.
    if not any(key in values for key in _RESULT_MARKER_KEYS):
        return None
    missing = [key for key in _RESULT_KEYS if key not in values]
    if missing:
        raise ValueError(f"incomplete solver result fields: missing={missing}")

    n_text = values["n"]
    residue_text = values["residue"]
    modulus_text = values["modulus"]
    wall_text = values["wall_s"]
    if not _UINT_RE.fullmatch(n_text):
        raise ValueError(f"non-canonical solver n: {n_text!r}")
    if not _UINT_RE.fullmatch(residue_text):
        raise ValueError(f"non-canonical solver residue: {residue_text!r}")
    if not _UINT_RE.fullmatch(modulus_text):
        raise ValueError(f"non-canonical solver modulus: {modulus_text!r}")
    if not _FLOAT_RE.fullmatch(wall_text):
        raise ValueError(f"invalid solver wall_s syntax: {wall_text!r}")

    n = int(n_text)
    residue = int(residue_text)
    modulus = int(modulus_text)
    wall_s = float(wall_text)
    if n < 1:
        raise ValueError(f"invalid solver n: {n}")
    if modulus < 2:
        raise ValueError(f"invalid solver modulus: {modulus}")
    if not math.isfinite(wall_s) or wall_s < 0:
        raise ValueError("solver wall_s must be finite and nonnegative")
    return SolverResult(n=n, residue=residue, modulus=modulus, wall_s=wall_s)
