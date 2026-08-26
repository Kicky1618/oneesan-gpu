#!/usr/bin/env python3
"""A/B v0.56 vs v0.57; wall_s is the retention metric.

v0.57 intentionally removes intermediate HIGH device waits, so the legacy
per-phase HIGH timers become enqueue-attribution rather than isolated device
phase times.  Do not compare those timers across this boundary.
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

base.VARIANTS.update({
    "v0.56": "src/cuda/b300/oneesan_b300_maskshard_v056_lowmaskbatch_runtime_tune.cu",
    "v0.57": "src/cuda/b300/oneesan_b300_maskshard_v057_highgroupsync.cu",
})
base.PHASE_KEYS = (
    "setup_s",
    "wall_s",
    "max_scratch_gib",
)
if "--variants" not in sys.argv[1:]:
    sys.argv[1:1] = ["--variants", "v0.56", "v0.57"]
raise SystemExit(base.main())
