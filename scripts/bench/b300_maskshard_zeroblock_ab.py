#!/usr/bin/env python3
"""A/B entrypoint isolating the v0.12 zero-BLOCKED-gather change.

Both candidates use the same v0.11 threshold-16 HIGH row packing and v0.9 LOW
closure mapping. The only intended execution difference is that v0.12 replaces
the row-boundary BLOCKED P2P gather with local zero initialization.
"""

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
        "v0.11": "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_fullclosure_highrowpack16_batch_guarded.cu",
        "v0.12": "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_fullclosure_highrowpack16_zeroblockgather_batch_guarded.cu",
    }
)

if "--variants" not in sys.argv[1:]:
    sys.argv[1:1] = ["--variants", "v0.11", "v0.12"]

raise SystemExit(base.main())
