#!/usr/bin/env python3
"""A/B v0.70 vs v0.71; compare residue identity, setup and wall time."""
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
    "v0.70": "src/cuda/b300/oneesan_b300_maskshard_v070_highperthreadstream.cu",
    "v0.71": "src/cuda/b300/oneesan_b300_maskshard_v071_highclosureclass.cu",
})
base.PHASE_KEYS = (
    "setup_s",
    "wall_s",
    "max_scratch_gib",
)
if "--variants" not in sys.argv[1:]:
    sys.argv[1:1] = ["--variants", "v0.70", "v0.71"]
raise SystemExit(base.main())
