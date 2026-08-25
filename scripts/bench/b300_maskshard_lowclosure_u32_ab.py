#!/usr/bin/env python3
"""A/B entrypoint isolating v0.40 uint32 LOW closure task arithmetic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE / "b300_maskshard_ab.py"

spec = importlib.util.spec_from_file_location("b300_maskshard_ab", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load base A/B driver: {BASE}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

base.VARIANTS.update(
    {
        "v0.39": "src/cuda/b300/oneesan_b300_maskshard_v039_lowgroup_packedcache.cu",
        "v0.40": "src/cuda/b300/oneesan_b300_maskshard_v040_lowclosure_u32.cu",
    }
)

if "--variants" not in sys.argv[1:]:
    sys.argv[1:1] = ["--variants", "v0.39", "v0.40"]

raise SystemExit(base.main())
