#!/usr/bin/env python3
"""A/B v0.66 vs v0.67; compare residue identity and wall_s."""
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
    "v0.66": "src/cuda/b300/oneesan_b300_maskshard_v066_highgroupsizeclass.cu",
    "v0.67": "src/cuda/b300/oneesan_b300_maskshard_v067_highrowplanclass.cu",
})
base.PHASE_KEYS = (
    "setup_s",
    "wall_s",
    "max_scratch_gib",
)
if "--variants" not in sys.argv[1:]:
    sys.argv[1:1] = ["--variants", "v0.66", "v0.67"]
raise SystemExit(base.main())
