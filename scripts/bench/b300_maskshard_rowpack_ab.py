#!/usr/bin/env python3
"""A/B entrypoint for v0.9/v0.10/v0.11 HIGH closure mappings.

This reuses the hardened base driver, including GPU-count/result identity checks,
sequential HBM use, raw logs, residue comparison, and build-provenance SHA-256
validation. If --variants is omitted, it compares v0.9, v0.10, and v0.11.
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
        "v0.10": "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_fullclosure_highrowpack_batch_guarded.cu",
        "v0.11": "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_fullclosure_highrowpack16_batch_guarded.cu",
    }
)

if "--variants" not in sys.argv[1:]:
    sys.argv[1:1] = ["--variants", "v0.9", "v0.10", "v0.11"]

raise SystemExit(base.main())
