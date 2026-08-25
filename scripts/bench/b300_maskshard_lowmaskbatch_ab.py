#!/usr/bin/env python3
"""A/B entrypoint for the LOW mask-batched executor.

Use v0.44 as the fair per-mask baseline because both candidates carry the
static-only LOW group cache.  The v0.43 batch experiment was developed before
v0.44 was numbered, so the version numbers are intentionally non-monotone in
this one comparison.
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
    "v0.44": "src/cuda/b300/oneesan_b300_maskshard_v044_lowgroup_staticcache.cu",
    "v0.43-batch": "src/cuda/b300/oneesan_b300_maskshard_v043_lowmaskbatch.cu",
})

# v0.43 reports the fused LOW phase as low_batch_sum_s instead of splitting
# orbit/closure timing.  Restrict the generic validator/table to fields common
# to both executors; residue equality and wall_s remain mandatory.
base.PHASE_KEYS = (
    "setup_s",
    "wall_s",
    "high_io_sum_s",
    "high_orbit_sum_s",
    "high_closure_sum_s",
    "max_scratch_gib",
)

if "--variants" not in sys.argv[1:]:
    sys.argv[1:1] = ["--variants", "v0.44", "v0.43-batch"]
raise SystemExit(base.main())
