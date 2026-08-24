#!/usr/bin/env python3
"""A/B entrypoint isolating v0.36 warp-row LOW orbit task mapping."""

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
        "v0.35": "src/cuda/b300/oneesan_b300_maskshard_v035_low_group_sync.cu",
        "v0.36": "src/cuda/b300/oneesan_b300_maskshard_v036_loworbit_warprow.cu",
    }
)

if "--variants" not in sys.argv[1:]:
    sys.argv[1:1] = ["--variants", "v0.35", "v0.36"]

raise SystemExit(base.main())
