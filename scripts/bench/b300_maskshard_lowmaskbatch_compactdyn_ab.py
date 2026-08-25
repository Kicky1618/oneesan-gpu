#!/usr/bin/env python3
"""A/B entrypoint isolating v0.47 compact resident LOW batch configs."""
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
    "v0.46": "src/cuda/b300/oneesan_b300_maskshard_v046_lowmaskbatch_ctacache.cu",
    "v0.47": "src/cuda/b300/oneesan_b300_maskshard_v047_lowmaskbatch_compactdyn.cu",
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
    sys.argv[1:1] = ["--variants", "v0.46", "v0.47"]
raise SystemExit(base.main())
