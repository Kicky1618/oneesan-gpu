#!/usr/bin/env python3
"""A/B/C entrypoint for LOW mask-batch CTA work targets."""
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

base.VARIANTS.update({
    "v0.50": "src/cuda/b300/oneesan_b300_maskshard_v050_lowmaskbatch_replica1024.cu",
    "v0.51": "src/cuda/b300/oneesan_b300_maskshard_v051_lowmaskbatch_target32768.cu",
    "v0.52": "src/cuda/b300/oneesan_b300_maskshard_v052_lowmaskbatch_target65536.cu",
})
base.PHASE_KEYS = (
    "setup_s",
    "wall_s",
    "high_io_sum_s",
    "high_orbit_sum_s",
    "high_closure_sum_s",
    "max_scratch_gib",
)
if "--variants" not in sys.argv[1:]:
    sys.argv[1:1] = ["--variants", "v0.50", "v0.51", "v0.52"]
raise SystemExit(base.main())
